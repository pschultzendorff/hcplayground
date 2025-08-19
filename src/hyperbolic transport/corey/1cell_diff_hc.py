import functools
import pathlib

import matplotlib.pyplot as plt
import numpy as np

results_dir = pathlib.Path(__file__).parent / "1cell_diff_hc_results"
results_dir.mkdir(exist_ok=True)


def f(s, mobility_ratio):
    return s**2 / (s**2 + mobility_ratio * (1 - s) ** 2)


def f_prime(s, mobility_ratio):
    return (2 * s * (1 - s) ** 2) / (s**2 + mobility_ratio * (1 - s) ** 2) ** 2


def f_double_prime(s, mobility_ratio):
    return (2 * (1 - s) ** 2 * (mobility_ratio * (1 - s) ** 2 - s**2)) / (
        s**2 + mobility_ratio * (1 - s) ** 2
    ) ** 3


def compute_adaptive_diffusion_coefficient(mobility_ratio, delta_t, omega):
    s_val = np.linspace(0, 1, 100)
    max_wave_speed = np.abs(f_prime(s_val, mobility_ratio)).max()
    return omega * delta_t * max_wave_speed


def g(s, mobility_ratio, adaptive_diffusion_coefficient, prev_s):
    return f(s, mobility_ratio) + adaptive_diffusion_coefficient * (s - prev_s)


def g_prime(s, mobility_ratio, adaptive_diffusion_coefficient):
    return f_prime(s, mobility_ratio) + adaptive_diffusion_coefficient


def g_double_prime(s, mobility_ratio):
    return f_double_prime(s, mobility_ratio)


def curvature(s, beta, mobility_ratio, delta_t, adaptive_diffusion_coefficient, prev_s):
    alpha = beta * g_prime(s, mobility_ratio, adaptive_diffusion_coefficient) + (
        1 - beta
    ) * f_prime(s, mobility_ratio)
    alpha_prime = beta * g_double_prime(s, mobility_ratio) + (
        1 - beta
    ) * f_double_prime(s, mobility_ratio)
    gamma = f(s, mobility_ratio) - g(
        s, mobility_ratio, adaptive_diffusion_coefficient, prev_s
    )
    a = 2 * delta_t**2 * gamma**2 / (1 + delta_t * alpha) ** 2
    b = delta_t**3 * alpha_prime * gamma**2 / (1 + delta_t * alpha) ** 3
    return np.abs(a + b)


def plot(mobility_ratio, delta_t, adaptive_diffusion_coefficient, prev_s):
    s_vals = np.linspace(0, 1, 100)
    beta_vals = np.linspace(0, 1, 100)

    B, S = np.meshgrid(beta_vals, s_vals)

    Z = np.vectorize(
        functools.partial(
            curvature,
            mobility_ratio=mobility_ratio,
            delta_t=delta_t,
            adaptive_diffusion_coefficient=adaptive_diffusion_coefficient,
            prev_s=prev_s,
        )
    )(S, B)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(Z, extent=(0, 1, 0, 1), origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Curvature")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$s$")
    ax.set_title("Curvature(s, beta)")

    return fig


for omega in [1e-5, 1e-3, 1e-1, 1]:
    sub_results_dir = results_dir / f"omega_{omega}"
    sub_results_dir.mkdir(exist_ok=True)
    for prev_s in [0.0, 0.5, 1.0]:
        for delta_t in [0.01, 0.1, 1.0, 10.0]:
            for mobility_ratio in [0.1, 1.0, 10.0]:
                adaptive_diffusion_coefficient = compute_adaptive_diffusion_coefficient(
                    mobility_ratio, delta_t, omega=omega
                )
                fig = plot(
                    mobility_ratio, delta_t, adaptive_diffusion_coefficient, prev_s
                )
                fig.savefig(
                    sub_results_dir
                    / f"curvature_plot_prev_s_{prev_s}_mobility_ratio_{mobility_ratio}_dt_{delta_t}.png",
                    bbox_inches="tight",
                )
                plt.close(fig)
