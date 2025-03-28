import torch
from thop import profile
import time
from typing import Dict, Any

# Base configuration for GPT models
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

# Model configurations for different GPT sizes
MODEL_CONFIGS = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Theoretical peak FLOPS per second for different GPU models
FLOPS_PER_SECOND = {
    "A100": 312e12,  # 312 TFLOPS for A100 GPU
    "V100": 125e12,  # 125 TFLOPS for V100 GPU
    "T4": 65e12,    # 65 TFLOPS for T4 GPU
    "P100": 21e12   # 21 TFLOPS for P100 GPU
}

def get_gpu_model(flops_per_second_dict: Dict[str, float]) -> str:
    """Identify the GPU model being used."""
    if not torch.cuda.is_available():
        return "CPU"
    
    device_name = torch.cuda.get_device_name(0)
    for model in flops_per_second_dict.keys():
        if model in device_name:
            return model
    return "Unknown"

def calculate_flops(model: torch.nn.Module, config: Dict[str, Any], batch_size: int = 1) -> float:
    """Calculate FLOPs for a given model configuration and batch size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create input tensor
    input_tensor = torch.randint(
        0, config["vocab_size"],
        (batch_size, config["context_length"]),
        device=device
    )
    
    # Calculate MACs and convert to FLOPs
    macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return 2 * macs  # Convert MACs to FLOPs (2 operations per MAC)

def measure_throughput(model: torch.nn.Module, config: Dict[str, Any], 
                      batch_size: int, num_iterations: int = 10) -> tuple[float, float]:
    """Measure model throughput and calculate Model FLOP Utilization (MFU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Warmup
    for _ in range(3):
        input_tensor = torch.randint(
            0, config["vocab_size"],
            (batch_size, config["context_length"]),
            device=device
        )
        model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        input_tensor = torch.randint(
            0, config["vocab_size"],
            (batch_size, config["context_length"]),
            device=device
        )
        model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    tokens_per_second = (batch_size * config["context_length"] * num_iterations) / elapsed_time
    
    # Calculate FLOPs
    flops = calculate_flops(model, config, batch_size)
    flops_per_token = flops / (batch_size * config["context_length"])
    achieved_flops = flops_per_token * tokens_per_second
    
    # Calculate MFU
    gpu_model = get_gpu_model(FLOPS_PER_SECOND)
    if gpu_model in FLOPS_PER_SECOND:
        mfu = achieved_flops / FLOPS_PER_SECOND[gpu_model]
    else:
        mfu = 0.0  # Cannot calculate MFU for unknown GPU
    
    return tokens_per_second, mfu

def find_max_batch_size(model_class: type, config: Dict[str, Any], 
                       start_batch: int = 1, max_batch: int = 4096) -> int:
    """Find the maximum batch size that fits in GPU memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_batch_size = start_batch
    max_possible_batch_size = max_batch
    
    while min_batch_size <= max_possible_batch_size:
        batch_size = (min_batch_size + max_possible_batch_size) // 2
        try:
            # Try to create and run model with current batch size
            model = model_class(config).bfloat16().to(device)
            input_tensor = torch.randint(
                0, config["vocab_size"],
                (batch_size, config["context_length"]),
                device=device
            )
            model(input_tensor)
            
            # If successful, try a larger batch size
            min_batch_size = batch_size + 1
            del model, input_tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                max_possible_batch_size = batch_size - 1
                try:
                    del model, input_tensor
                except NameError:
                    pass
                torch.cuda.empty_cache()
            else:
                raise e
    
    return max_possible_batch_size