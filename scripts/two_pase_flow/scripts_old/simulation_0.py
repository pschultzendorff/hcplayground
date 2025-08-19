import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from hc import DiffusionHC, FluxHC4, FluxHC5  # noqa: E402;
from model import TPFModel  # noqa: E402
from plotting import solve_and_plot  # noqa: E402

results_dir = pathlib.Path(__file__).parent / "simulation_0_results"
results_dir.mkdir(exist_ok=True)

NUM_CELLS = 500
DOMAIN_LENGTH = 1.0

CELL_SIZE = DOMAIN_LENGTH / NUM_CELLS


model_params = {
    "num_cells": NUM_CELLS,
    "transmissibilities": jnp.full((NUM_CELLS + 1,), CELL_SIZE),  # Includes
    "porosities": jnp.full((NUM_CELLS,), 1.0),
    "pe": 0.0,  # Only viscous effects.
    "source_terms": jnp.full((NUM_CELLS, 2), 0.0),
    "bc": ["neumann", "dirichlet"],
    "neumann_bc": jnp.array(
        [
            [0.01, 0.01],  # Saturation at Neumann boundary is 1.
            [0.0, 0.0],
        ]
    ),
    "dirichlet_bc": jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    ),
    "initial_conditions": jnp.stack(
        [
            jnp.zeros((NUM_CELLS,)),
            jnp.concatenate(
                [jnp.full((NUM_CELLS // 2,), 0.01), jnp.full((NUM_CELLS // 2,), 0.99)]
            ),
        ],
        axis=1,
    ).flatten(),
    "mu_w": 1.0,  # Viscosity ratio of \mu_n / \mu_w = 5
    "mu_n": 5.0,
}


class CoreyRelPerms(TPFModel):
    def mobility_w(self, s):
        return s**2 / self.mu_w

    def mobility_n(self, s):
        return (1 - s) ** 2 / self.mu_n


class FluxHC4CoreyRelPerms(CoreyRelPerms, FluxHC4): ...


class FluxHC5CoreyRelPerms(CoreyRelPerms, FluxHC5): ...


class DiffusionHC4CoreyRelPerms(CoreyRelPerms, DiffusionHC):
    def update_adaptive_diffusion_coeff(self, dt, x_prev):
        pass


model_fluxhc1 = FluxHC4CoreyRelPerms(**model_params, k=30.0)
model_fluxhc2 = FluxHC5CoreyRelPerms(**model_params, k=30.0)
model_diffhc1 = DiffusionHC4CoreyRelPerms(**model_params, adaptive_diffusion_coeff=0.14)

final_time = 5.0

curvature_fig, curvature_ax = plt.subplots(figsize=(10, 6))
curvature_ax2 = curvature_ax.twinx()
for i, (model, color) in enumerate(
    zip(
        [
            model_fluxhc1,
            model_fluxhc2,
            model_diffhc1,
        ],
        sns.color_palette("rocket", 3),
    ),
    start=1,
):
    solution_curve_fig, curvature_fig = solve_and_plot(
        model,
        final_time,
        color,
        f"{model.__class__.__name__} {i}",
        curvature_fig=curvature_fig,
        max_iter=10,
        appleyard=True,
    )
    if solution_curve_fig is not None:
        solution_curve_fig.savefig(
            results_dir / f"solution_curve_HC_{i}_T_{final_time}.png",
            bbox_inches="tight",
        )
        plt.close(solution_curve_fig)
if curvature_fig is not None:
    curvature_fig.savefig(
        results_dir / f"curvature_T_{final_time}.png",
        bbox_inches="tight",
    )
    plt.close(curvature_fig)
    plt.close(curvature_fig)
