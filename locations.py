"""
Details about the local filesystem.

This should be the only file that needs to be changed when using a new machine.
"""

# Directory for each dataset.
mnist = "./mnist"
cifar10 = "./cifar10"
imagenet = "./imagenet"

# Directory containing accurate networks for each dataset, to be used for
# knowledge distillation.
teachers = "./teachers"
