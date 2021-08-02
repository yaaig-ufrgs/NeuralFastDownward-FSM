"""
Simple auxiliary functions.
"""

SAMPLE_INIT_STATE = 1
SAMPLE_RANDOM_STATE = 2
SAMPLE_ENTIRE_PLAN = 3


def to_unary(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]
