import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

sns.set_theme(style="whitegrid")


def plot_solution(solutions, plot_pw=False, model=None):
    """Plot the solution of the two-phase flow problem with an interactive time slider."""
    # Convert solutions to numpy array for easier manipulation.
    solutions_array = jnp.array(solutions)
    n_time_steps = len(solutions)

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()  # Create a second y-axis for the saturation.
    plt.subplots_adjust(bottom=0.25)  # Make room for slider.

    # Cell centers for plotting.
    xx = jnp.linspace(0, solutions.shape[-1] // 2, solutions.shape[-1] // 2)

    # Initial plot with first time step.
    pressures = solutions_array[0, ::2]
    saturations = solutions_array[0, 1::2]
    if plot_pw:
        if model is None:
            raise ValueError("Model must be provided for wetting pressure calculation.")
        wetting_pressures = solutions_array[:, ::2] - model.pc(solutions_array[:, 1::2])
    else:
        wetting_pressures = jnp.zeros_like(solutions_array[:, ::2])

    (pn_line,) = ax.plot(xx, pressures, "o-", color="tab:blue", label=r"$p_n$")
    (pw_line,) = ax.plot(
        xx, wetting_pressures[0], "x-", color="tab:green", label=r"$p_w$"
    )
    (s_line,) = ax2.plot(xx, saturations, "v-", color="tab:orange", label=r"$s_w$")

    # Find min and max pressure values for consistent y-axis.
    p_min = min(
        jnp.concatenate([solutions_array[:, ::2], wetting_pressures]).min(),  # type: ignore
        0,
    )
    p_max = jnp.concatenate([solutions_array[:, ::2], wetting_pressures]).max() * 1.1
    ax.set_ylim(p_min, p_max)  # type: ignore

    ax2.set_ylim(0, 1)  # Saturation is between 0 and 1.

    ax.set_xlabel("x")
    ax.set_ylabel(r"$p_n$", color="tab:blue")
    ax2.set_ylabel(r"$s_w$", color="tab:orange")

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


def plot_curvature_distance(betas, curvatures=None, distances=None, fig=None, **kwargs):
    """Plot the curvature of the homotopy curve."""
    betas = jnp.asarray(betas)

    # Create figure and axis.
    if fig is None:
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax2 = ax1.twinx()
    else:
        ax1, ax2 = fig.axes

    # Plot curvature and distance over \lambda range.
    if curvatures is not None:
        curvatures = jnp.asarray(curvatures)
        ax1.plot(betas, curvatures, **kwargs)
    if distances is not None:
        distances = jnp.asarray(distances)
        ax2.plot(betas, distances, **kwargs)

    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$\kappa$")
    ax2.set_ylabel(
        r"$\frac{\|\mathbf{x}_{\lambda=0} - \mathbf{x}_{\lambda=1}\|}{\|\mathbf{x}_{\lambda=0}\|}$"
    )
    ax1.set_xlim(1, 0)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_title("Curvature and weighted distance to exact solution")
    ax1.grid(True)
    ax1.legend()
    ax2.legend()

    return fig


def plot_solution_curve(solutions, betas):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 6), subplot_kw={"projection": "3d"}
    )

    solutions = jnp.asarray(solutions)

    X = jnp.asarray(betas)
    Y = jnp.linspace(0, solutions.shape[-1] // 2, solutions.shape[-1] // 2)
    X, Y = jnp.meshgrid(X, Y)  # type: ignore

    p = jnp.asarray(solutions)[:, ::2].swapaxes(0, 1)
    s = jnp.asarray(solutions)[:, 1::2].swapaxes(0, 1)

    ax1.plot_surface(X, Y, p, cmap="coolwarm", edgecolor="none", label=r"$p_n$")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("Cell index")
    ax1.set_zlabel(r"$p_n$")
    ax1.set_xlim(1, 0)
    ax1.set_ylim(
        p.shape[0] - 1, 0
    )  # Reverse y-axis for cell index, since pressure is decreasing througout the domain.
    ax1.view_init(elev=30, azim=-30)
    ax1.set_title("Solution curves")

    ax2.plot_surface(X, Y, s, cmap="plasma", edgecolor="none", label=r"$s_w$")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Cell index")
    ax2.set_zlabel(r"$s_w$")
    ax2.set_xlim(1, 0)
    ax2.view_init(elev=30, azim=-30)

    return fig
