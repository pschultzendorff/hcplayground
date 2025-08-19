import pathlib
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from hc import DiffusionHC, FluxHC4, FluxHC5, solve  # noqa: E402; solve,
from model import TPFModel  # noqa: E402
from plotting import solve_and_plot  # noqa: E402

results_dir = pathlib.Path(__file__).parent / "30cells_7_results"
results_dir.mkdir(exist_ok=True)

NUM_CELLS = 30

model_params = {
    "num_cells": NUM_CELLS,
    "domain_size": 1.0,
    "permeabilities": jnp.full((NUM_CELLS,), 1.0),
    "porosities": jnp.full((NUM_CELLS,), 0.5),
    "pe": jnp.concatenate(
        [
            jnp.full((NUM_CELLS // 2,), 0.5),
            jnp.full((NUM_CELLS // 2,), 10.0),
        ]
    ),
    "source_terms": jnp.full((NUM_CELLS, 2), 0.0),
    "bc": ["neumann", "dirichlet"],
    "neumann_bc": jnp.array(
        [
            [1.0, 0.0],
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
    "mu_w": 5.0,
    "mu_n": 1.0,
}

# model = TPFModel(**model_params)
# solutions, _ = solve(model, final_time=1, n_time_steps=50)
# plot_solution(solutions)

# model = TPFModel(p_e=30.0)
# solutions, _ = solve(model, final_time=10.0, n_time_steps=100, tol=5e-5)
# plot_solution(solutions, plot_pw=True, model=model)


model_fluxhc1 = FluxHC4(**model_params, k=30.0)
model_fluxhc2 = FluxHC5(**model_params, k=30.0)
model_diffhc1 = DiffusionHC(**model_params, fixed_diffusion_coeff=1e-5)
model_diffhc2 = DiffusionHC(**model_params, fixed_diffusion_coeff=1e-3)
model_diffhc3 = DiffusionHC(**model_params, fixed_diffusion_coeff=1)

for final_time in [0.5, 1.0, 10.0, 50.0]:
    curvature_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for i, (model, color) in enumerate(
        zip(
            [
                model_fluxhc1,
                model_fluxhc2,
                model_diffhc1,
                model_diffhc2,
                model_diffhc3,
            ],
            sns.color_palette("rocket", 5),
        ),
        start=1,
    ):
        solution_curve_fig, residual_curve_fig, curvature_fig = solve_and_plot(
            model,
            final_time,
            color,
            f"{model.__class__.__name__} {i}",
            curvature_fig=curvature_fig,
        )
        if solution_curve_fig is not None:
            solution_curve_fig.savefig(
                results_dir / f"solution_curve_HC_{i}_T_{final_time}.png",
                bbox_inches="tight",
            )
            plt.close(solution_curve_fig)
        if residual_curve_fig is not None:
            residual_curve_fig.savefig(
                results_dir / f"residual_curve_HC_{i}_T_{final_time}.png",
                bbox_inches="tight",
            )
            plt.close(residual_curve_fig)
    if curvature_fig is not None:
        curvature_fig.savefig(
            results_dir / f"curvature_T_{final_time}.png",
            bbox_inches="tight",
        )
        plt.close(curvature_fig)
