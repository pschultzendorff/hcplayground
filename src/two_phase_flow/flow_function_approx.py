"""This module approximates the flow function for two-phase flow using a rational
function. It allows for visualization and adjustment of the mobility ratio through
interactive sliders.

"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.optimize import minimize

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from numeric_function_typing import numeric_binary_func
from numpy.typing import ArrayLike as ArrayLike_np


@numeric_binary_func
def f_w(s, mobility_ratio):
    """Flow function :math:`f_w(s) = s^2 / (s^2 + M * (1 - s)^2)`."""
    mobility_w = s**2
    mobility_t = s**2 + mobility_ratio * (1 - s) ** 2
    return mobility_w / mobility_t


def rational_gn_with_pn0(
    s: float | np.ndarray, p_coef: ArrayLike_np, q_coef: ArrayLike_np
) -> np.ndarray:
    """Rational function :math:`g_n(x) = P_n(x) / Q_n(x)` with :math:`P_n(0) = 0`."""
    p_coef_full = np.copy(p_coef)
    # Force constant term zero to ensure g(0) = 0.
    p_coef_full[-1] = 0
    p = np.polyval(p_coef_full, s)
    # Mypy complains about ArrayLike_np not being _ArraLikeComplex_co.
    q = np.polyval(q_coef, s)  # type: ignore
    return p / q


def objective(
    params: ArrayLike_np, ss: np.ndarray, ys_true: np.ndarray, deg_num: int
) -> np.ndarray:
    """Objective function for fitting rational function to the flow function."""
    p_coef = np.asarray(params)[: deg_num + 1]
    q_coef = np.asarray(params)[deg_num + 1 :]
    ys_pred = rational_gn_with_pn0(ss, p_coef, q_coef)
    return np.max((ys_true - ys_pred) ** 2)


def concavity_convexity_constraint(
    params: ArrayLike_np, ss: np.ndarray, deg_num: int
) -> np.ndarray:
    """Constraint to ensure the rational function is neither concave nor convex."""
    p_coef = np.asarray(params)[: deg_num + 1]
    q_coef = np.asarray(params)[deg_num + 1 :]
    ys_pred = rational_gn_with_pn0(ss, p_coef, q_coef)
    # Second derivative.
    dg_ds = np.gradient(ys_pred, ys_pred[1] - ss[0])
    d2g_ds2 = np.abs(np.gradient(dg_ds, ss[1] - ss[0]))
    return d2g_ds2 - 1e-3


def monotonicity_constraint_bc(
    params: ArrayLike_np, ss: np.ndarray, deg_num: int
) -> np.ndarray:
    """Constraint to ensure the rational function is monotone."""
    p_coef = np.asarray(params)[: deg_num + 1]
    q_coef = np.asarray(params)[deg_num + 1 :]
    ys_pred = rational_gn_with_pn0(ss, p_coef, q_coef)
    dg_ds = np.gradient(ys_pred, ss[1] - ss[0])
    return dg_ds


def boundary_constraint(params: ArrayLike_np, deg_num: int):
    """Constraint to ensure the rational function takes value 1 at 1."""
    p_coef = np.asarray(params)[: deg_num + 1]
    q_coef = np.asarray(params)[deg_num + 1 :]
    p_coef_full = np.copy(p_coef)
    p_coef_full[-1] = 0  # Enforce p_n=0.
    P1 = np.polyval(p_coef_full, 1)
    Q1 = np.polyval(q_coef, 1)
    return P1 - Q1


def combined_constraints_bc(
    params: ArrayLike_np, ss: np.ndarray, deg_num: int
) -> np.ndarray:
    """Combine the convexity, concavity, and monotonicity constraints."""
    return np.concatenate(
        [
            concavity_convexity_constraint(params, ss, deg_num),
            monotonicity_constraint_bc(params, ss, deg_num),
        ]
    )


def fit_rational_function(ss: np.ndarray, ys: np.ndarray, deg_num=3, deg_den=3):
    """Fit a rational function to the flow function under the given constraints."""
    init_params = np.random.randn(deg_num + 1 + deg_den + 1)
    cons = [
        {
            "type": "ineq",
            "fun": lambda params: combined_constraints_bc(params, ss, deg_num),
        },
        {
            "type": "eq",
            "fun": lambda params: boundary_constraint(params, deg_num),
        },
    ]

    result = minimize(
        objective,
        init_params,
        args=(ss, ys, deg_num),
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
    )

    if not result.success:
        print("Warning: optimization failed:", result.message)
        return np.full_like(ss, np.nan)

    p_coef = result.x[: deg_num + 1]
    q_coef = result.x[deg_num + 1 :]
    return rational_gn_with_pn0(ss, p_coef, q_coef)


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[2, 1])
    plt.subplots_adjust(bottom=0.25)

    ss = np.linspace(0.001, 0.999, 300)
    mobility_ratio = 1.0

    ys_true = f_w(ss, mobility_ratio)
    ys_fit = fit_rational_function(ss, ys_true)
    residual = ys_true - ys_fit

    # Plot initial curves.
    (line_f,) = ax1.plot(ss, ys_true, label="f(x)", lw=2)
    (line_g,) = ax1.plot(ss, ys_fit, label="g(x): rational", lw=2, linestyle="--")
    ax1.legend()
    ax1.set_title("Rational Approximation to the Flow Function")
    ax1.grid()

    (line_r,) = ax2.plot(ss, residual, label="Residual f(x) - g(x)", color="red")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.legend()
    ax2.grid()

    # Add sliders for mobility ratio.
    ax_mr = plt.axes((0.15, 0.1, 0.65, 0.03))
    slider_mr = Slider(ax_mr, "M", 0.1, 5.0, valinit=mobility_ratio, valstep=0.1)

    def update(val):
        mobility_ratio_new = slider_mr.val
        ys_new = f_w(ss, mobility_ratio_new)
        ys_fit_new = fit_rational_function(ss, ys_new)
        residual_new = ys_new - ys_fit_new

        line_f.set_ydata(ys_new)
        line_g.set_ydata(ys_fit_new)
        line_r.set_ydata(residual_new)

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw_idle()

    slider_mr.on_changed(update)

    plt.show()
