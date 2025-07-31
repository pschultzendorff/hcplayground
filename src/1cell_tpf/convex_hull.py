from typing import Any, Callable, Literal, overload

import jax.numpy as jnp
import numpy as np

HullSide = Literal["upper", "lower"]
NdArray = np.ndarray | jnp.ndarray


def update_array(array: NdArray, idx: int | tuple[Any] | list[int] | NdArray, value):
    """Update array at index `idx` with `value`. Works for both numpy and jax.numpy."""
    if isinstance(array, np.ndarray):
        array[idx] = value
        return array
    elif isinstance(array, jnp.ndarray):
        # Indexing with lists does not work in jax.numpy, so we convert lists to arrays.
        if isinstance(idx, list):
            idx = jnp.array(idx)
        return array.at[idx].set(value)
    else:
        raise TypeError(
            f"Unsupported array type: {type(array)}. Expected numpy or jax.numpy."
        )


def andrews_monotone_chain(
    f: Callable,
    interval: tuple[float, float],
    side: HullSide,
    num_points: int = 100,
    xp=np,
) -> tuple[NdArray, NdArray]:
    """
    Computes the convex hull of a set of points using the Andrews monotone chain algorithm.

    Parameters:
        f: A function that takes an x values and returns a y value.
        interval: A tuple containing the start and end of the interval to consider.
        side: Whether to return the upper or lower convex hull.
        num_points: The number of points to sample along the interval. Defaults to 100.
        xp: The numpy-like module to use (e.g., `numpy` or `jax.numpy`). Defaults to
            `numpy`.

    Returns:
        points: ``shape=(num_points, 2)`` A :obj:`~numpy.ndarray` of sample points
            along the interval.
        mask: ``shape=(num_points,)`` A :obj:`~numpy.ndarray` of boolean values
            indicating which sample points are part of the convex hull.

    """
    xx: NdArray = xp.linspace(interval[0], interval[1], num_points)
    yy: NdArray = xp.vectorize(f)(xx)
    # Stack to create an array of points on the graph of the function. No need to sort
    # or filter out duplicates, as xx was generated sorted and with unique values.
    points: NdArray = xp.column_stack((xx, yy))

    # Initialize the convex hull with the first two points.
    hull: list[NdArray] = list(points[:2])

    # Compare the two last points with the next point in the list. If they form a right
    # turn, remove the last point from the lower hull. If they form a left turn, add the
    # new point to the lower hull. Vice versa for the upper hull.
    turn_direction = 1 if side == "lower" else -1
    for c in points[2:]:
        while (
            len(hull) >= 2
            and turn_direction * xp.cross(hull[-1] - hull[-2], hull[-1] - c) >= 0
        ):
            hull.pop()
        hull.append(c)

    hull_array: NdArray = xp.array(hull)

    # Create a boolean mask to filter points that are part of the convex hull.
    mask: NdArray = xp.zeros((num_points,), dtype=bool)
    for point in points:
        if xp.any((hull_array == point).all(axis=1)):
            mask = update_array(mask, xp.nonzero((points == point).all(axis=1)), True)

    return points, mask


@overload
def convex_hull(
    f: Callable,
    interval: tuple[float, float],
    side: HullSide,
    f_prime: None,
    f_double_prime: None,
    xp,
    **kwargs: Any,
) -> Callable: ...


@overload
def convex_hull(
    f: Callable,
    interval: tuple[float, float],
    side: HullSide,
    f_prime: Callable,
    f_double_prime: None,
    xp,
    **kwargs: Any,
) -> tuple[Callable, Callable]: ...


@overload
def convex_hull(
    f: Callable,
    interval: tuple[float, float],
    side: HullSide,
    f_prime: None,
    f_double_prime: Callable,
    xp,
    **kwargs: Any,
) -> tuple[Callable, Callable]: ...


@overload
def convex_hull(
    f: Callable,
    interval: tuple[float, float],
    side: HullSide,
    f_prime: Callable,
    f_double_prime: Callable,
    xp,
    **kwargs: Any,
) -> tuple[Callable, Callable, Callable]: ...


def convex_hull(
    f: Callable,
    interval: tuple[float, float],
    side: HullSide,
    f_prime: Callable | None = None,
    f_double_prime: Callable | None = None,
    xp=np,
    **kwargs: Any,
):
    """Returns the convex hull of a function (and its derivatives) as a new function.

    Parameters:
        f: A function that takes an x value and returns a y value.
        interval: A tuple containing the start and end of the interval to consider.
        side: Whether to return the upper or lower convex hull.
        f_prime: Optional. The first derivative of the function.
        f_double_prime: Optional. The second derivative of the function.
        xp: The numpy-like module to use (e.g., `numpy` or `jax.numpy`). Defaults to
            `numpy`.
        **kwargs: Additional keyword arguments to pass to the `andrews_monotone_chain`
            function.

    Returns:
        A tuple of functions representing the convex hull, and optionally its first and
        second derivatives if provided.

    """
    points, mask = andrews_monotone_chain(f, interval, side, xp=xp, **kwargs)
    points_on_hull: NdArray = points[mask]

    def f_hull(x: float | NdArray) -> NdArray:
        """Returns the y value of the convex hull at a given x value."""
        x = xp.asarray(x)
        if xp.any(x < interval[0]) or xp.any(x > interval[1]):
            raise ValueError(f"x must be in the interval {interval}, got {x}.")

        # Find indices of the sample points closest to x (left neighbor, guaranteed
        # not equal to x). Clip to ensure idxs is not negative (if x equal the left
        # boundary of the interval).
        idxs = xp.searchsorted(points[:, 0], x) - 1
        idxs = xp.clip(idxs, 0, len(points) - 2)

        # If both neighbor points lie on the convex hull, return :math:`f'(x)`. If
        # either of the neighbors is not on the hull, return the linear interpolation
        # between the two points on the hull that are closest to x.
        hull_equals_function = mask[idxs] & mask[idxs + 1]
        hull_equals_interpolation = ~hull_equals_function

        return (
            f(x) * hull_equals_function
            + xp.interp(x, points_on_hull[:, 0], points_on_hull[:, 1])
            * hull_equals_interpolation
        )

    hull_functions = [f_hull]

    if f_prime is not None:

        def f_prime_hull(x: float | NdArray) -> NdArray:
            """Returns the first derivative of the convex hull at a given x value."""
            x = xp.asarray(x)
            if xp.any(x < interval[0]) or xp.any(x > interval[1]):
                raise ValueError(f"x must be in the interval {interval}, got {x}.")

            # Find indices of the sample points closest to x (left neighbor, guaranteed
            # not equal to x). Clip to ensure idxs is not negative (if x equal the left
            # boundary of the interval).
            idxs = xp.searchsorted(points[:, 0], x) - 1
            idxs = xp.clip(idxs, 0, len(points) - 2)

            # If both neighbor points lie on the convex hull, return :math:`f'(x)`. If
            # either of the neighbors is not on the hull, return the slope of the linear
            # segment of the hull.
            hull_equals_function = mask[idxs] & mask[idxs + 1]
            hull_equals_interpolation = ~hull_equals_function

            # To calculate the slope, find the two closest points on the hull.
            idxs_hull = xp.searchsorted(points_on_hull[:, 0], x) - 1
            idxs_hull = xp.clip(idxs_hull, 0, len(points_on_hull) - 2)

            linear_slope: NdArray = (
                points_on_hull[idxs_hull + 1, 1] - points_on_hull[idxs_hull, 1]
            ) / (points_on_hull[idxs_hull + 1, 0] - points_on_hull[idxs_hull, 0])

            return (
                f_prime(x) * hull_equals_function
                + linear_slope * hull_equals_interpolation
            )

        hull_functions.append(f_prime_hull)

    if f_double_prime is not None:

        def f_double_prime_hull(x: float | NdArray) -> NdArray:
            """Returns the second derivative of the convex hull at a given x value."""
            x = xp.asarray(x)
            if xp.any(x < interval[0]) or xp.any(x > interval[1]):
                raise ValueError(f"x must be in the interval {interval}, got {x}.")

            # Find indices of the sample points closest to x (left neighbor, guaranteed
            # not equal to x). Clip to ensure idxs is not negative (if x equal the left
            # boundary of the interval).
            idxs = xp.searchsorted(points[:, 0], x) - 1
            idxs = xp.clip(idxs, 0, len(points) - 2)

            # If both neighbor points lie on the convex hull, return :math:`f''(x)`. If
            # either of the neighbors is not on the hull, return the curvature of the
            # linear hull segment, i.e., 0.
            hull_equals_function = mask[idxs] & mask[idxs + 1]

            return f_double_prime(x) * hull_equals_function

        hull_functions.append(f_double_prime_hull)

    if len(hull_functions) == 1:
        return hull_functions[0]
    else:
        return tuple(hull_functions)
