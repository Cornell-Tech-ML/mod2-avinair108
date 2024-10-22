"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply two numbers (floats) together"""
    return x * y


def id(x: float) -> float:
    """Returns a given number back unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers (floats) to each other"""
    return x + y


def neg(x: float) -> float:
    """Returns the negation of the input number"""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one input number is less than the other"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two input numbers are equal"""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the greater of the two input numbers"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function of x"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Computes the ReLU of x"""
    return max(0.0, x)


def log(x: float) -> float:
    """Computes the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of arg"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg"""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return d if x > 0 else 0.0


def map(func: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Applies a given function to each element of an iterable.

    Args:
    ----
        func: A function that takes an element of type float and returns type float.
        iterable: An iterable of elements of type float.

    Returns:
    -------
        An iterable of elements of type float obtained by applying `func` to each element in `iterable`.

    """
    for item in iterable:
        yield func(item)


def zipWith(
    func: Callable[[float, float], float],
    iterable1: Iterable[float],
    iterable2: Iterable[float],
) -> Iterable[float]:
    """Combines elements from two iterables using a given function.

    Args:
    ----
        func: A function that takes two floats (one from each iterable) and combines them into a float.
        iterable1: The first iterable of elements of type float.
        iterable2: The second iterable of elements of type float.

    Returns:
    -------
        An iterable of combined floats obtained by applying `func` to pairs from `iterable1` and `iterable2`.

    """
    iter1, iter2 = iter(iterable1), iter(iterable2)
    while True:
        try:
            yield func(next(iter1), next(iter2))
        except StopIteration:
            break


def reduce(
    func: Callable[[float, float], float], iterable: Iterable[float], initial: float
) -> float:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        func: A function that combines two elements of the iterable into a float.
        iterable: An iterable of elements of type float.
        initial: The initial value to start the reduction.

    Returns:
    -------
        A single float obtained by repeatedly applying `func` to the elements of `iterable`.

    """
    result = initial
    iterator = iter(iterable)
    while True:
        try:
            item = next(iterator)
            result = func(result, item)
        except StopIteration:
            break
    return result


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates each element in the input list."""
    return map(lambda x: -x, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two lists."""
    return zipWith(lambda x, y: x + y, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sums all elements in the list."""
    return reduce(lambda x, y: x + y, lst, 0.0)


def prod(lst: Iterable[float]) -> float:
    """Takes the product of all elements in the list."""
    return reduce(lambda x, y: x * y, lst, 1.0)
