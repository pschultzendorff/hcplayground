"""Visualization for homotopy curves of the 1D two-phase flow problem."""

from typing import Optional, TypeAlias

import jax.numpy as jnp
import seaborn as sns
from hc import HCModel
from jax.typing import ArrayLike as ArrayLike_jax
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from numpy.typing import ArrayLike as ArrayLike_np
from solvers import solve

from hcplayground.src.two_phase_flow.model_tpf import TPFModel

ArrayLike: TypeAlias = ArrayLike_jax | ArrayLike_np

sns.set_theme(style="whitegrid")


def plot_solution(
    solutions: ArrayLike, plot_pw: bool = False, model: Optional[TPFModel] = None
):
    """Plot the solution of the two-phase flow problem with an interactive time slider.

    Parameters:
        solutions: Array-like structure containing the solution data.
        plot_pw: Whether to plot wetting pressure. Defaults to False.
        model: Optional TPFModel instance for calculating wetting pressure. Defaults to
        None.

    """
    sol = jnp.array(solutions)
    n_time_steps, n_vars = sol.shape
    n_cells = n_vars // 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    plt.subplots_adjust(bottom=0.25)  # Make room for slider.

    xs = jnp.linspace(0, n_cells, n_cells)  # Cell centers for plotting.

    pressures = sol[:, ::2]
    saturations = sol[:, 1::2]
    if plot_pw:
        if model is None:
            raise ValueError("Model must be provided for wetting pressure calculation.")
        pressures_w = pressures - model.pc(pressures)
    else:
        pressures_w = jnp.zeros_like(pressures)

    (pn_line,) = ax.plot(xs, pressures, "o-", color="tab:blue", label="$p_n$")
    (pw_line,) = ax.plot(xs, pressures_w, "x-", color="tab:green", label=r"$p_w$")
    (s_line,) = ax2.plot(xs, saturations, "v-", color="tab:orange", label="$s_w$")

    arr = jnp.concatenate([pressures, pressures_w])
    ax.set_ylim(min(arr.min(), 0), arr.max() * 1.1)  # type: ignore
    ax2.set_ylim(0, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("$p_n$", color="tab:blue")
    ax2.set_ylabel("$s_w$", color="tab:orange")

    ax.set_title("Solution (time step: 0)")
    ax.legend()
    ax2.legend()
    ax.grid(True)

    # Add slider
    ax_slider = plt.axes((0.15, 0.1, 0.7, 0.03))
    time_slider = Slider(ax_slider, r"$t$", 0, n_time_steps, valinit=0, valstep=1)

    # Update function for slider
    def update(val):
        time_idx = int(time_slider.val)
        pn_line.set_ydata(pressures[time_idx])
        pw_line.set_ydata(pressures_w[time_idx])
        s_line.set_ydata(saturations[time_idx])
        ax.set_title(f"Solution (time step: {time_idx})")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def weighted_curvature(
    curvature_vectors: ArrayLike, betas: ArrayLike, intermediate_solutions: ArrayLike
):
    r"""Calculated the curvature weighted by total arclength.

    We use equations (30) and (31) from Brown & Zingg (2017) to approximate the
    arclength of individual curve segments and the entire curve.

    .. math::
        \Delta s_i \approx \sqrt{\|\mathbf{x}_i - \mathbf{x}_{i - 1}\|^2 + \|\beta_i - \beta_{i - 1}\|^2}.

    .. math::
        s_{tot} \approx \sum_{i=1}^{n} \Delta s_i.


    Brown, D.A. and Zingg, D.W. (2017) ‘Design and evaluation of homotopies for
    efficient and robust continuation’, Applied Numerical Mathematics, 118, pp. 150–181.
    Available at: https://doi.org/10.1016/j.apnum.2017.03.001.

    """
    curvatures = jnp.linalg.norm(jnp.asarray(curvature_vectors), axis=-1)
    betas = jnp.asarray(betas)
    intermediate_solutions = jnp.asarray(intermediate_solutions)

    # Calculate the total arclength of the homotopy curve.
    segments_arclengths = jnp.sqrt(
        jnp.linalg.norm(
            intermediate_solutions[1:] - intermediate_solutions[:-1], axis=-1
        )
        ** 2
        + (betas[1:] - betas[:-1]) ** 2
    )
    total_arclength = jnp.sum(segments_arclengths, axis=0)

    return curvatures * total_arclength**2


def weighted_distance(approximations: ArrayLike, exact_solution: jnp.ndarray):
    """Weight distance in pressure and saturations separately by the magnitude of the
    exact solution."""
    approximations = jnp.asarray(approximations)

    pressure_norm = jnp.linalg.norm(exact_solution[::2])
    saturation_norm = jnp.linalg.norm(exact_solution[1::2])

    distances = (
        jnp.linalg.norm(
            approximations[:, ::2] - exact_solution[None, ...][:, ::2],
            axis=-1,
        )
        / pressure_norm
        + jnp.linalg.norm(
            approximations[:, 1::2] - exact_solution[None, ...][:, 1::2],
            axis=-1,
        )
        / saturation_norm
    )

    return distances


def plot_curvature_and_distance(
    betas: ArrayLike,
    curvatures: ArrayLike | None = None,
    distances: ArrayLike | None = None,
    fig: Optional[Figure] = None,
    **kwargs,
):
    """Plot the curvature of the homotopy curve."""
    betas = jnp.asarray(betas)

    # Create figure and axis.
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        ax1, ax2 = fig.axes

    # Plot curvature and distance over beta range.
    if curvatures is not None:
        ax1.plot(betas, jnp.asarray(curvatures), linewidth=2, **kwargs)
    if distances is not None:
        ax2.plot(betas, jnp.asarray(distances), linewidth=2, **kwargs)

    ax1.set_xlabel(r"$\beta$", fontsize=12)
    ax1.set_ylabel(r"$Curvature \kappa$", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_xlim(1, 0)
    # ax1.set_ylim(bottom=1e-3, top=1e8)
    ax1.set_title("Curvature Components Along Homotopy Path", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel(r"$\beta$", fontsize=12)
    ax2.set_ylabel(r"$\tilde{d}(\mathbf{x}_{\beta = 1}, \mathbf{x}_{\beta = 0})$")
    ax2.set_yscale("log")
    ax2.set_xlim(1, 0)
    # ax2.set_ylim(bottom=1e-3, top=1)
    ax2.set_title(r"Weighted Distance to Solution", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.5)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    for legend in fig.legends:
        legend.remove()
    # Make space for the legend.
    # fig.subplots_adjust(right=0.75)
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        fontsize=10,
        frameon=False,
    )

    fig.suptitle("Homotopy Curvature and Relative Error", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.75, 0.95))

    return fig


def plot_solution_curve(solutions: ArrayLike, betas: ArrayLike, model: HCModel):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
    )

    solutions = jnp.asarray(solutions)

    X = jnp.asarray(betas)
    Y = jnp.linspace(0, model.domain_size, model.num_cells)
    X, Y = jnp.meshgrid(X, Y)

    # Pressure plot.
    surf1 = ax1.plot_surface(
        X,
        Y,
        solutions[:, ::2].swapaxes(0, 1),
        cmap="viridis",
        edgecolor="k",
        linewidth=0.2,
    )
    ax1.set_xlabel(r"$\beta$", fontsize=12, labelpad=10)
    ax1.set_ylabel("$x$", fontsize=12, labelpad=10)
    ax1.set_zlabel("$p_n$", fontsize=12, labelpad=10)
    ax1.set_xlim(1, 0)
    ax1.view_init(elev=30, azim=-45)
    ax1.set_title("Nonwetting Pressure $p_n$", fontsize=14)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.15)

    # Saturation plot.
    surf2 = ax2.plot_surface(
        X,
        Y,
        solutions[:, 1::2].swapaxes(0, 1),
        cmap="plasma",
        edgecolor="k",
        linewidth=0.2,
    )
    ax2.set_xlabel(r"$\beta$", fontsize=12, labelpad=10)
    ax2.set_ylabel("$x$", fontsize=12, labelpad=10)
    ax2.set_zlabel("$s_w$", fontsize=12, labelpad=10)
    ax2.set_xlim(1, 0)
    ax2.view_init(elev=30, azim=-45)
    ax2.set_title("Water Saturation $s_w$", fontsize=14)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.15)

    fig.suptitle("Solution Curves Along Homotopy Path", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_residual_curve(
    solutions: ArrayLike,
    betas: ArrayLike,
    model: HCModel,
    dt: float,
    x_prev: Optional[jnp.ndarray] = None,
):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
    )

    solutions = jnp.asarray(solutions)

    # Compute residuals w.r.t. :math:`H(\cdot,\beta=0) = F(\cdot)`.
    beta_save = model.beta
    model.beta = 0.0
    residuals = jnp.abs(
        jnp.asarray(
            [model.residual(solution, dt, x_prev=x_prev) for solution in solutions]
        )
    )

    model.beta = beta_save

    X = jnp.asarray(betas)
    Y = jnp.linspace(0, model.domain_size, model.num_cells)
    X, Y = jnp.meshgrid(X, Y)

    # Flow residuals.
    surf1 = ax1.plot_surface(
        X,
        Y,
        residuals[:, : model.num_cells].swapaxes(0, 1),
        cmap="viridis",
        edgecolor="k",
        linewidth=0.2,
    )
    ax1.set_xlabel(r"$\beta$", fontsize=12, labelpad=10)
    ax1.set_ylabel("$x$", fontsize=12, labelpad=10)
    ax1.set_zlabel(r"$\mathcal{R}_{\mathrm{flow}}$", fontsize=12, labelpad=10)
    ax1.set_xlim(1, 0)
    ax1.view_init(elev=30, azim=-45)
    ax1.set_title("Cellwise Flow Residuals", fontsize=14)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.15)

    # Transport residuals.
    surf2 = ax2.plot_surface(
        X,
        Y,
        residuals[:, model.num_cells :].swapaxes(0, 1),
        cmap="plasma",
        edgecolor="k",
        linewidth=0.2,
    )
    ax2.set_xlabel(r"$\beta$", fontsize=12, labelpad=10)
    ax2.set_ylabel("$x$", fontsize=12, labelpad=10)
    ax2.set_zlabel(r"$\mathcal{R}_{\mathrm{transport}}$", fontsize=12, labelpad=10)
    ax2.set_xlim(1, 0)
    ax2.view_init(elev=30, azim=-45)
    ax2.set_title("Cellwise Transport Residuals", fontsize=14)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.15)

    fig.suptitle("Residuals Along Homotopy Path", fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig


def solve_and_plot(
    model: HCModel,
    final_time: float,
    color: str,
    solver_name: str,
    curvature_fig: Optional[Figure] = None,
    **kwargs,
):
    model.reset()

    _, converged = solve(model, final_time=final_time, n_time_steps=1, **kwargs)
    intermediate_solutions = jnp.asarray(model.intermediate_solutions)

    if converged:
        print(
            f"{solver_name}: Relative distance between solutions:"
            + f" {jnp.linalg.norm(intermediate_solutions[-1] - intermediate_solutions[0]) / jnp.linalg.norm(intermediate_solutions[-1])}"
        )
        print(
            f"{solver_name}: Relative distance between pressure solutions:"
            + f" {jnp.linalg.norm(intermediate_solutions[-1][::2] - intermediate_solutions[0][::2]) / jnp.linalg.norm(intermediate_solutions[-1][::2])}"
        )
        print(
            f"{solver_name}: Relative distance between saturation solutions:"
            + f" {jnp.linalg.norm(intermediate_solutions[-1][1::2] - intermediate_solutions[0][1::2]) / jnp.linalg.norm(intermediate_solutions[-1][1::2])}"
        )
        distances = weighted_distance(
            intermediate_solutions[:-1], intermediate_solutions[-1]
        )
        curvature_fig = plot_curvature_and_distance(
            model.betas[:-1],
            distances=distances,
            fig=curvature_fig,
            label=rf"$\tilde{{d}}(\mathbf{{x}}_{{\beta = 1}}, \mathbf{{x}}_{{\beta = 0}})$ ({solver_name})",
            color=color,
            ls=":",
        )
    if len(model.betas) > 1:
        solution_curve_fig = plot_solution_curve(
            intermediate_solutions, model.betas, model
        )
        residual_curve_fig = plot_residual_curve(
            intermediate_solutions,
            model.betas,
            model,
            final_time,
            model.initial_conditions,
        )

        # Calculate and plot weighted curvatures as a traceability measure.
        full_weighted_curvature = weighted_curvature(
            model.curvature_vectors,
            model.betas,
            intermediate_solutions,
        )
        curvature_fig = plot_curvature_and_distance(
            model.betas,
            curvatures=full_weighted_curvature,
            fig=curvature_fig,
            label=rf"$s_{{tot}}^2 \kappa$ ({solver_name})",
            color=color,
        )
        pressure_weighted_curvature = weighted_curvature(
            jnp.asarray(model.curvature_vectors)[:, ::2],
            model.betas,
            intermediate_solutions[:, ::2],
        )
        curvature_fig = plot_curvature_and_distance(
            model.betas,
            curvatures=pressure_weighted_curvature,
            fig=curvature_fig,
            label=rf"$s_{{tot,p}}^2 \kappa_p$ ({solver_name})",
            color=color,
            ls="--",
        )
        saturation_weighted_curvature = weighted_curvature(
            jnp.asarray(model.curvature_vectors)[:, 1::2],
            model.betas,
            intermediate_solutions[:, 1::2],
        )
        curvature_fig = plot_curvature_and_distance(
            model.betas,
            curvatures=saturation_weighted_curvature,
            fig=curvature_fig,
            label=rf"$s_{{tot,s}}^2 \kappa_s$ ({solver_name})",
            color=color,
            ls="-.",
        )

        # Reset again to empty data lists and avoid memory issues.
        model.reset()

        return solution_curve_fig, residual_curve_fig, curvature_fig
    else:
        model.reset()
        return None, None, curvature_fig
