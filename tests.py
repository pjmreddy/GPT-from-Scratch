

from gpt import main

expected = """
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      OUT
==================================================

Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
Output length: 15
Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""


def test_main(capsys):
    main()
    captured = capsys.readouterr()

    # Normalize line endings and strip trailing whitespace from each line
    normalized_expected = '\n'.join(line.rstrip() for line in expected.splitlines())
    normalized_output = '\n'.join(line.rstrip() for line in captured.out.splitlines())

    # Compare normalized strings
    assert normalized_output == normalized_expected

if __name__ == "__main__": 
    test_main("Hello, I am")  # Run the test if this script is executed directly
    print("Test passed!")  # Print a success message if the test passes
