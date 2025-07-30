import math
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from convex_hull import (
    HullSide,
    NdArray,
    andrews_monotone_chain,
    convex_hull,
    update_array,
)


@pytest.fixture(params=[np, jnp])
def xp(request: pytest.FixtureRequest):
    """Run tests for both numpy and jax.numpy."""
    return request.param


@pytest.fixture
def function(xp, request: pytest.FixtureRequest) -> Callable:
    if request.param == "parabola":
        return lambda x: x**2
    elif request.param == "sine":
        return lambda x: xp.sin(x)
    else:
        raise ValueError(f"Unknown function: {request.param}")


@pytest.fixture
def function_prime(xp, request: pytest.FixtureRequest) -> Callable:
    if request.param == "parabola":
        return lambda x: 2 * x
    elif request.param == "sine":
        return lambda x: xp.cos(x)
    else:
        raise ValueError(f"Unknown function: {request.param}")


@pytest.fixture
def function_double_prime(xp, request: pytest.FixtureRequest) -> Callable:
    if request.param == "parabola":
        return lambda x: xp.full_like(x, 2.0)
    elif request.param == "sine":
        return lambda x: -xp.sin(x)
    else:
        raise ValueError(f"Unknown function: {request.param}")


@pytest.mark.parametrize("side", ["upper", "lower"])
@pytest.mark.parametrize("num_points", [30, 100, 500])
@pytest.mark.parametrize(
    "function, interval",
    [
        ("parabola", (-1, 1)),
        ("sine", (0, 2.5 * math.pi)),
        ("sine", (math.pi / 2, 3 * math.pi)),
    ],
    indirect=["function"],
)
def test_andrews_monotone_chain(
    function: Callable,
    interval: tuple[float, float],
    side: HullSide,
    num_points: int,
    xp,
):
    """Test :meth:`andrews_monotone_chain` with sine and parabola on various intervals."""
    points, mask = andrews_monotone_chain(
        function, interval, side, num_points=num_points, xp=xp
    )

    # Validate output shapes and types.
    assert points.shape == (num_points, 2)
    assert mask.shape == (num_points,)
    assert mask.dtype == bool

    # Validate that the first and last sample points are on the convex hull.
    assert mask[0]
    assert mask[-1]

    # Validate that the upper and lower hulls differ.
    if side == "upper":
        _, mask_lower = andrews_monotone_chain(function, interval, "lower", xp=xp)
        assert not xp.all(mask == mask_lower)


@pytest.mark.parametrize("side", ["upper", "lower"])
@pytest.mark.parametrize(
    "function, function_prime, function_double_prime, interval",
    [
        ("parabola",) * 3 + ((-1, 1),),
        ("sine",) * 3 + ((0, 2.5 * math.pi),),
    ],
    indirect=["function", "function_prime", "function_double_prime"],
)
def test_convex_hull(
    function: Callable,
    function_prime: Callable,
    function_double_prime: Callable,
    interval: tuple[float, float],
    side: HullSide,
    xp,
):
    """Test convex hull evaluation matches or bounds the original function."""

    (f_hull, f_prime_hull, f_double_prime_hull) = convex_hull(
        function,
        interval,
        side,
        f_prime=function_prime,
        f_double_prime=function_double_prime,
        xp=xp,
    )
    xs: NdArray = xp.linspace(interval[0], interval[1], 100)
    ys: NdArray = function(xs)
    ys_prime: NdArray = function_prime(xs)
    ys_double_prime: NdArray = function_double_prime(xs)
    ys_hull: NdArray = f_hull(xs)
    ys_hull_prime: NdArray = f_prime_hull(xs)
    ys_hull_double_prime: NdArray = f_double_prime_hull(xs)

    # Validate that the functions return values' shapes match the inputs' shapes.
    assert ys_hull.shape == xs.shape
    assert ys_hull_prime.shape == xs.shape
    assert ys_hull_double_prime.shape == xs.shape

    # Validate that values are finite.
    assert xp.all(xp.isfinite(ys_hull))
    assert xp.all(xp.isfinite(ys_hull_prime))
    assert xp.all(xp.isfinite(ys_hull_double_prime))

    # The lower hull should not be above the function, and the upper hull should not be
    # below the function.
    if side == "lower":
        assert xp.all(ys_hull <= ys)
    elif side == "upper":
        assert xp.all(ys_hull >= ys)

    # Verify convex hull evaluation at interval endpoints match the original function.
    assert xp.isclose(f_hull(interval[0]), function(interval[0]))
    assert xp.isclose(f_hull(interval[1]), function(interval[1]))

    # Validate that the hull's derivatives match the function's derivatives where the
    # hull and the function coincide. At the endpoints, the hull and functions always
    # coincide, so we don't check here.
    hull_equals_function_mask = xp.isclose(ys_hull, ys)
    hull_equals_function_mask = update_array(hull_equals_function_mask, [0, -1], False)  # type: ignore
    assert xp.all(
        xp.isclose(
            ys_hull_prime[hull_equals_function_mask],
            ys_prime[hull_equals_function_mask],
        )
    )
    assert xp.all(
        xp.isclose(
            ys_hull_double_prime[hull_equals_function_mask],
            ys_double_prime[hull_equals_function_mask],
        )
    )

    # Validate that the hull's second derivative vanishes where the hull and function
    # deviate.
    assert xp.all(xp.isclose(ys_hull_double_prime[~hull_equals_function_mask], 0.0))


@pytest.mark.parametrize(
    "function, function_prime, function_double_prime",
    [("parabola",) * 3],
    indirect=True,
)
def test_convex_hull_raises_for_out_of_bounds(
    function: Callable, function_prime: Callable, function_double_prime: Callable, xp
):
    """Verify convex hull raises ValueError for points outside the interval."""
    (f_hull, f_hull_prime, f_hull_double_prime) = convex_hull(
        function,
        (-1, 1),
        "lower",
        f_prime=function_prime,
        f_double_prime=function_double_prime,
        xp=xp,
    )

    for hull_function in [f_hull, f_hull_prime, f_hull_double_prime]:
        for x in [-2, math.pi]:
            with pytest.raises(ValueError):
                hull_function(x)
