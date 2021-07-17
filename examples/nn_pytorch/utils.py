"""
Simple auxiliary functions.
"""

def to_unary(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]
