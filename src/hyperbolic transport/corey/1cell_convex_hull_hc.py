import functools
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from convex_hull import convex_hull  # noqa: E402

results_dir = pathlib.Path(__file__).parent / "1cell_convex_hull_hc_results"
results_dir.mkdir(exist_ok=True)


def f(s, mobility_ratio):
    return s**2 / (s**2 + mobility_ratio * (1 - s) ** 2)


def f_prime(s, mobility_ratio):
    return (2 * s * (1 - s) ** 2) / (s**2 + mobility_ratio * (1 - s) ** 2) ** 2


def f_double_prime(s, mobility_ratio):
    return (2 * (1 - s) ** 2 * (mobility_ratio * (1 - s) ** 2 - s**2)) / (
        s**2 + mobility_ratio * (1 - s) ** 2
    ) ** 3


def curvature(s, beta, mobility_ratio, delta_t):
    alpha = beta * g_prime(s) + (1 - beta) * f_prime(s, mobility_ratio)
    alpha_prime = beta * g_double_prime(s) * (1 - beta) + f_double_prime(
        s, mobility_ratio
    )
    gamma = f(s, mobility_ratio) - g(s)
    a = 2 * delta_t**2 * gamma**2 / (1 + delta_t * alpha) ** 2
    b = delta_t**3 * alpha_prime * gamma**2 / (1 + delta_t * alpha) ** 3
    return np.abs(a + b)


def plot(mobility_ratio, delta_t):
    s_vals = np.linspace(0, 1, 100)
    beta_vals = np.linspace(0, 1, 100)

    B, S = np.meshgrid(beta_vals, s_vals)

    Z = np.vectorize(
        functools.partial(curvature, mobility_ratio=mobility_ratio, delta_t=delta_t)
    )(S, B)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(
        Z,
        extent=(0, 1, 0, 1),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Curvature")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$s$")
    ax.set_title("Curvature(s, beta)")

    return fig


for delta_t in [0.01, 0.1, 1.0, 10.0]:
    for mobility_ratio in [0.1, 1.0, 10.0]:
        # For mobility_ratio < 1.0, the wetting flow resembles a concave function and we
        # construct the upper convex hull.
        # For mobility_ratio >= 1.0, the wetting flow resembles a convex function and we
        # construct the lower convex hull.
        side = "upper" if mobility_ratio < 1.0 else "lower"
        g, g_prime, g_double_prime = convex_hull(
            functools.partial(f, mobility_ratio=mobility_ratio),
            [0, 1],
            side,
            f_prime=functools.partial(f_prime, mobility_ratio=mobility_ratio),
            f_double_prime=functools.partial(
                f_double_prime, mobility_ratio=mobility_ratio
            ),
        )

        fig = plot(mobility_ratio, delta_t)
        fig.savefig(
            results_dir
            / f"curvature_plot_mobility_ratio_{mobility_ratio}_dt_{delta_t}.png",
            bbox_inches="tight",
        )
        plt.close(fig)
