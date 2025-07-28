import jax.numpy as jnp
import seaborn as sns
from hc import solve
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

sns.set_theme(style="whitegrid")


def plot_solution(solutions, plot_pw=False, model=None):
    """Plot the solution of the two-phase flow problem with an interactive time slider."""
    # Convert solutions to numpy array for easier manipulation.
    solutions_array = jnp.array(solutions)
    n_time_steps = solutions_array.shape[0]
    n_cells = solutions_array.shape[-1] // 2

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()  # Create a second y-axis for the saturation.
    plt.subplots_adjust(bottom=0.25)  # Make room for slider.

    # Cell centers for plotting.
    xx = jnp.linspace(0, n_cells, n_cells)

    # Initial plot with first time step.
    pressures = solutions_array[0, ::2]
    saturations = solutions_array[0, 1::2]
    if plot_pw:
        if model is None:
            raise ValueError("Model must be provided for wetting pressure calculation.")
        wetting_pressures = solutions_array[:, ::2] - model.pc(solutions_array[:, 1::2])
    else:
        wetting_pressures = jnp.zeros_like(solutions_array[:, ::2])

    (pn_line,) = ax.plot(xx, pressures, "o-", color="tab:blue", label="$p_n$")
    (pw_line,) = ax.plot(
        xx, wetting_pressures[0], "x-", color="tab:green", label=r"$p_w$"
    )
    (s_line,) = ax2.plot(xx, saturations, "v-", color="tab:orange", label="$s_w$")

    # Find min and max pressure values for consistent y-axis.
    p_min = min(
        jnp.concatenate([solutions_array[:, ::2], wetting_pressures]).min(),  # type: ignore
        0,
    )
    p_max = jnp.concatenate([solutions_array[:, ::2], wetting_pressures]).max() * 1.1
    ax.set_ylim(p_min, p_max)  # type: ignore

    ax2.set_ylim(0, 1)  # Saturation is between 0 and 1.

    ax.set_xlabel("x")
    ax.set_ylabel("$p_n$", color="tab:blue")
    ax2.set_ylabel("$s_w$", color="tab:orange")

    ax.set_title("Solution (time step: 0)")
    ax.legend()
    ax2.legend()
    ax.grid(True)

    # Add slider
    ax_slider = plt.axes((0.15, 0.1, 0.7, 0.03))
    time_slider = Slider(
        ax=ax_slider,
        label=r"$t$",
        valmin=0,
        valmax=n_time_steps,
        valinit=0,
        valstep=1,
    )

    # Update function for slider
    def update(val):
        time_idx = int(time_slider.val)
        pn_line.set_ydata(solutions_array[time_idx, ::2])
        pw_line.set_ydata(wetting_pressures[time_idx])
        s_line.set_ydata(solutions_array[time_idx, 1::2])
        ax.set_title(f"Solution (time step: {time_idx})")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def weighted_distance(approximations, exact_solution):
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
    betas, curvatures=None, distances=None, fig=None, **kwargs
):
    """Plot the curvature of the homotopy curve."""
    betas = jnp.asarray(betas)

    # Create figure and axis.
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        ax1, ax2 = fig.axes

    # Plot curvature and distance over \beta range.
    if curvatures is not None:
        curvatures = jnp.asarray(curvatures)
        ax1.plot(betas, curvatures, linewidth=2, **kwargs)
    if distances is not None:
        distances = jnp.asarray(distances)
        ax2.plot(betas, distances, linewidth=2, **kwargs)

    ax1.set_xlabel(r"$\beta$", fontsize=12)
    ax1.set_ylabel(r"$Curvature \kappa$", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_xlim(1, 0)
    ax1.set_ylim(bottom=1e-3, top=1e8)
    ax1.set_title("Curvature Components Along Homotopy Path", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel(r"$\beta$", fontsize=12)
    ax2.set_ylabel(r"$\tilde{d}(\mathbf{x}_{\beta = 1}, \mathbf{x}_{\beta = 0})$")
    ax2.set_yscale("log")
    ax2.set_xlim(1, 0)
    ax2.set_ylim(bottom=1e-3, top=1)
    ax2.set_title(r"Weighted Distance to Solution at $\beta = 0$", fontsize=14)
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


def plot_solution_curve(solutions, betas, model, diffusion_coeff=None):
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

    # Optional diffusion coefficient overlay
    # if diffusion_coeff is not None:
    #     X_overlay =jnp.array([1])
    #     Y_overlay =jnp.linspace(0, solutions.shape[-1] // 2, solutions.shape[-1] // 2)
    #     X_overlay, Y_overlay =jnp.meshgrid(X_overlay, Y_overlay)

    #     for ax in [ax1, ax2]:
    #         ax_overlay = fig.add_axes(ax.get_position(), projection="3d")
    #         ax_overlay.plot(
    #             X_overlay,
    #             Y_overlay,
    #             diffusion_coeff[:-1],
    #             color="red",
    #             linewidth=2,
    #             label=r"$\\beta$",
    #         )
    #         ax_overlay.set_xlim(ax.get_xlim())
    #         ax_overlay.set_ylim(ax.get_ylim())
    #         ax_overlay.set_xticks([])
    #         ax_overlay.set_yticks([])
    #         ax_overlay.set_zticks([])
    #         ax_overlay.xaxis.line.set_color((0, 0, 0, 0))
    #         ax_overlay.yaxis.line.set_color((0, 0, 0, 0))
    #         ax_overlay.view_init(elev=30, azim=-45)

    fig.suptitle("Solution Curves Along Homotopy Path", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_residual_curve(solutions, betas, model, dt, x_prev=None):
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


def solve_and_plot(model, final_time, color, solver_name, curvature_fig=None, **kwargs):
    model.reset()

    _, converged = solve(model, final_time=final_time, n_time_steps=1, **kwargs)
    intermediate_solutions = jnp.asarray(model.intermediate_solutions)
    curvature_vectors = jnp.asarray(model.curvature_vectors)

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
        # Plot solution curve AND diffusion coefficients for diffusion based HC.
        # if isinstance(model, DiffusionHC):
        #     solution_curve_fig = plot_solution_curve(
        #         model.intermediate_solutions,
        #         model.betas,
        #         model.adaptive_diffusion_coeff,
        #     )
        # else:
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

        curvature_fig = plot_curvature_and_distance(
            model.betas,
            curvatures=jnp.linalg.norm(curvature_vectors, axis=-1),
            fig=curvature_fig,
            label=rf"$\kappa$ ({solver_name})",
            color=color,
        )
        curvature_fig = plot_curvature_and_distance(
            model.betas,
            curvatures=jnp.linalg.norm(curvature_vectors[:, ::2], axis=-1),
            fig=curvature_fig,
            label=rf"$\kappa_p$ ({solver_name})",
            color=color,
            ls="--",
        )
        curvature_fig = plot_curvature_and_distance(
            model.betas,
            curvatures=jnp.linalg.norm(curvature_vectors[:, 1::2], axis=-1),
            fig=curvature_fig,
            label=rf"$\kappa_s$ ({solver_name})",
            color=color,
            ls="-.",
        )

        # Reset again to empty data lists and avoid memory issues.
        model.reset()

        return solution_curve_fig, residual_curve_fig, curvature_fig
    else:
        model.reset()
        return None, None, curvature_fig
