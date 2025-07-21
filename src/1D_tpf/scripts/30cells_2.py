import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from hc import (  # noqa: E402
    DiffusionHC,
    FluxHC1,
    FluxHC2,
    FluxHC3,
    FluxHC4,
    FluxHC5,
    solve,
)
from model import TPFModel  # noqa: E402
from plotting import (  # noqa: E402
    plot_curvature_distance,
    plot_solution,
    plot_solution_curve,
    weighted_distance,
)

results_dir = pathlib.Path(__file__).parent / "30cells_2_results"
results_dir.mkdir(exist_ok=True)

key = jax.random.PRNGKey(0)  # Reproducability.
NUM_CELLS = 30
RAND = jax.random.uniform(key, shape=(NUM_CELLS + 1,), minval=-1.0, maxval=2.0)

TRANSMISSIBILITIES = jnp.pow(10, RAND)
TRANSMISSIBILITIES = TRANSMISSIBILITIES.at[0].set(0.0)
TRANSMISSIBILITIES = TRANSMISSIBILITIES.at[-1].set(TRANSMISSIBILITIES[-1] * 2)
# Transmissibility at the Dirichlet boundary has to be doubled.
# Transmissibility at the Neumann boundary has to be set to 0.

POROSITIES = (RAND + 2.5) / 5.0  # Scale to [0.1, 0.5] range.
POROSITIES = POROSITIES[1:]  # Porosities are cellwise, not facewise.


model_params = {
    "num_cells": 2,
    "transmissibilities": TRANSMISSIBILITIES,
    "porosities": POROSITIES,
    "source_terms": jnp.full((NUM_CELLS, 2), 0.0),
    "neumann_bc": jnp.array(
        [
            [1.0, 0.1],
            [0.0, 0.0],
        ]
    ),
    "dirichlet_bc": jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.98],
        ]
    ),
    "initial_conditions": jnp.stack(
        [
            jnp.zeros((NUM_CELLS,)),
            jnp.concatenate(
                [jnp.full((NUM_CELLS // 2,), 0.02), jnp.full((NUM_CELLS // 2,), 0.98)]
            ),
        ],
        axis=1,
    ).flatten(),
    "mu_w": 1.0,
    "mu_n": 5.0,
}

# model = TPFModel(**model_params, p_e=2.0)
# solutions, _ = solve(model, final_time=1.0, n_time_steps=100)
# plot_solution(solutions)

# model = TPFModel(p_e=30.0)
# solutions, _ = solve(model, final_time=10.0, n_time_steps=100, tol=5e-5)
# plot_solution(solutions, plot_pw=True, model=model)


def solve_and_plot(model, final_time, color, curvature_fig=None, i=1):
    model.reset()
    _, converged = solve(model, final_time=final_time, n_time_steps=1)
    if converged:
        # plot_solution(model_fluxhc1.intermediate_solutions)
        print(
            f"{model.__class__.__name__} {i}: Relative distance between solutions:"
            + f" {jnp.linalg.norm(model.intermediate_solutions[-1] - model.intermediate_solutions[0]) / jnp.linalg.norm(model.intermediate_solutions[-1])}"
        )
        print(
            f"{model.__class__.__name__} {i}: Relative distance between pressure solutions:"
            + f" {jnp.linalg.norm(model.intermediate_solutions[-1][::2] - model.intermediate_solutions[0][::2]) / jnp.linalg.norm(model.intermediate_solutions[-1][::2])}"
        )
        print(
            f"{model.__class__.__name__} {i}: Relative distance between saturation solutions:"
            + f" {jnp.linalg.norm(model.intermediate_solutions[-1][1::2] - model.intermediate_solutions[0][1::2]) / jnp.linalg.norm(model.intermediate_solutions[-1][1::2])}"
        )
    if len(model.betas) > 1:
        solution_curve_fig = plot_solution_curve(
            model.intermediate_solutions, model.betas
        )
        curvature_fig = plot_curvature_distance(
            model.betas,
            curvatures=jnp.linalg.norm(jnp.asarray(model.curvature_vectors), axis=-1),
            fig=curvature_fig,
            label=rf"$\kappa$ ({model.__class__.__name__} {i})",
            color=color,
            marker="o",
        )
        curvature_fig = plot_curvature_distance(
            model.betas,
            curvatures=jnp.linalg.norm(
                jnp.asarray(model.curvature_vectors)[:, ::2], axis=-1
            ),
            fig=curvature_fig,
            label=rf"$\kappa_p$ ({model.__class__.__name__} {i})",
            color=color,
            ls="--",
            marker="v",
        )
        curvature_fig = plot_curvature_distance(
            model.betas,
            curvatures=jnp.linalg.norm(
                jnp.asarray(model.curvature_vectors)[:, 1::2], axis=-1
            ),
            fig=curvature_fig,
            label=rf"$\kappa_s$ ({model.__class__.__name__} {i})",
            color=color,
            ls="-.",
            marker="x",
        )
        intermediate_solutions = jnp.asarray(model.intermediate_solutions)
        distances = weighted_distance(
            intermediate_solutions[:-1], intermediate_solutions[-1]
        )
        curvature_fig = plot_curvature_distance(
            model.betas[:-1],
            distances=distances,
            fig=curvature_fig,
            label=rf"$\frac{{\|\mathbf{{x}}_{{\lambda=0}} - \mathbf{{x}}_{{\lambda=1}}\|}}{{\|\mathbf{{x}}_{{\lambda=0}}\|}}$ ({model.__class__.__name__} {i})",
            color=color,
            ls=":",
            marker="^",
        )
        return solution_curve_fig, curvature_fig
    else:
        return None, curvature_fig


color_palette_fluxhc = sns.color_palette("rocket", 6)
color_palette_diffhc = sns.color_palette("mako", 6)

for p_e in [0.05, 2.0, 10.0, 50]:
    model_fluxhc1 = FluxHC1(**model_params, p_e=p_e)
    model_fluxhc2 = FluxHC2(**model_params, p_e=p_e)
    model_fluxhc3 = FluxHC3(**model_params, p_e=p_e)
    model_fluxhc4 = FluxHC4(**model_params, p_e=p_e, k=10.0)
    model_fluxhc5 = FluxHC5(**model_params, p_e=p_e, k=10.0)
    model_fluxhc6 = FluxHC5(**model_params, p_e=p_e, k=30.0)
    model_diffhc1 = DiffusionHC(**model_params, p_e=p_e, kappa=0.01)
    model_diffhc2 = DiffusionHC(**model_params, p_e=p_e, kappa=0.1)
    model_diffhc3 = DiffusionHC(**model_params, p_e=p_e, kappa=1.0)
    model_diffhc4 = DiffusionHC(**model_params, p_e=p_e, kappa=10.0)
    model_diffhc5 = DiffusionHC(**model_params, p_e=p_e, kappa=0.001)
    model_diffhc6 = DiffusionHC(**model_params, p_e=p_e, kappa=100.0)

    for final_time in [0.5, 1.0, 10.0, 50.0]:
        curvature_fig, curvature_ax = plt.subplots(figsize=(10, 6))
        curvature_ax2 = curvature_ax.twinx()
        for i, (model_fluxhc, color) in enumerate(
            zip(
                [
                    model_fluxhc1,
                    model_fluxhc2,
                    model_fluxhc3,
                    model_fluxhc4,
                    model_fluxhc5,
                    model_fluxhc6,
                ],
                color_palette_fluxhc,
            ),
            start=1,
        ):
            solution_curve_fig, curvature_fig = solve_and_plot(
                model_fluxhc, final_time, color, curvature_fig=curvature_fig, i=i
            )
            if solution_curve_fig is not None:
                solution_curve_fig.savefig(
                    results_dir
                    / f"solution_curve_flux_hc_{i}_T_{final_time}_pe_{p_e}.png",
                    bbox_inches="tight",
                )
                plt.close(solution_curve_fig)
        if curvature_fig is not None:
            curvature_fig.savefig(
                results_dir / f"curvature_flux_hc_T_{final_time}_pe_{p_e}.png",
                bbox_inches="tight",
            )
            plt.close(curvature_fig)

        curvature_fig, curvature_ax = plt.subplots(figsize=(10, 6))
        curvature_ax2 = curvature_ax.twinx()
        for i, (model_diffhc, color) in enumerate(
            zip(
                [
                    model_diffhc1,
                    model_diffhc2,
                    model_diffhc3,
                    model_diffhc4,
                    model_diffhc5,
                    model_diffhc6,
                ],
                color_palette_diffhc,
            ),
            start=1,
        ):
            solution_curve_fig, curvature_fig = solve_and_plot(
                model_diffhc, final_time, color, curvature_fig=curvature_fig, i=i
            )
            if solution_curve_fig is not None:
                solution_curve_fig.savefig(
                    results_dir
                    / f"solution_curve_diff_hc_{i}_T_{final_time}_pe_{p_e}.png",
                    bbox_inches="tight",
                )
                plt.close(solution_curve_fig)
        if curvature_fig is not None:
            curvature_fig.savefig(
                results_dir / f"curvature_diff_hc_T_{final_time}_pe_{p_e}.png",
                bbox_inches="tight",
            )
            plt.close(curvature_fig)
