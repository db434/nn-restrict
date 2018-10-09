import sys


def info(*args, **kwargs):
    """Report low-priority information. Has the same interface as `print`."""
    print(*args, file=sys.stderr, **kwargs)


def error(*args, **kwargs):
    """Report high-priority information. Has the same interface as `print`."""
    print(*args, file=sys.stderr, **kwargs)
