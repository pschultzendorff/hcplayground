from typing import Callable, Protocol, TypeAlias, cast, overload

import numpy as np

NumericInput: TypeAlias = float | np.ndarray


class NumericUnaryFunc(Protocol):
    """Reusable protocol for numeric functions that have a (float)->float and
    (array)->array signature.

    """

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray: ...


def numeric_unary_func(f: Callable) -> NumericUnaryFunc:
    """Decorator to mark functions that adhere to NumericUnaryFunc protocol."""
    return cast(NumericUnaryFunc, f)


class NumericBinaryFunc(Protocol):
    """Reusable protocol for numeric functions that have a (float,float)->float and
    (array|float,array|float)->array signature.

    """

    @overload
    def __call__(self, x: float, y: float) -> float: ...
    @overload
    def __call__(self, x: np.ndarray, y: float) -> np.ndarray: ...
    @overload
    def __call__(self, x: float, y: np.ndarray) -> np.ndarray: ...
    @overload
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


def numeric_binary_func(f: Callable) -> NumericBinaryFunc:
    """Decorator to mark functions that adhere to NumericBinaryFunc protocol."""
    return cast(NumericBinaryFunc, f)


@numeric_binary_func
def f(x, y):
    return x * y


class A:
    @numeric_binary_func
    def multiply(self, x, y):
        return f(x, y)
