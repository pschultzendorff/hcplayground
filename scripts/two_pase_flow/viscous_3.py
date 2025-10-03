import pathlib
import sys
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

parent_dir = pathlib.Path(__file__).parent.parent.parent / "src" / "two_phase_flow"
sys.path.append(str(parent_dir))

from hc import (
    ConvexHullFluxHC,
    DiffusionHC,
    LinearCapPressHC,
    LinearRelPermHC,
    ZeroCapPressHC,
)
from viz import solve_and_plot

results_dir = pathlib.Path(__file__).parent / "viscous_3"
results_dir.mkdir(exist_ok=True)

NUM_CELLS: int = 30

model_params: dict[str, Any] = {
    "num_cells": NUM_CELLS,
    "domain_size": 1.0,
    "permeabilities": jnp.full((NUM_CELLS,), 1e-6),
    "porosities": jnp.full((NUM_CELLS,), 0.1),
    "source_terms": jnp.full((NUM_CELLS, 2), 0.0),
    "bc": ["neumann", "dirichlet"],
    "dirichlet_bc": jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.98],
        ]
    ),
    "rp_model": "Corey",
    "cp_model": "zero",
}


class FluxHC1(LinearCapPressHC, LinearRelPermHC): ...


def compare_solvers(model_params: dict[str, Any], run_case: str) -> None:
    case_dir = results_dir / run_case
    case_dir.mkdir(exist_ok=True)

    # Determine the convex hull side based on the structure of the Riemann problem.
    match run_case:
        case "case1_s02_s98_u0_mu5_mu1":
            hull_side = "lower"
        case "case2_s98_s02_u1_mu5_mu1":
            hull_side = "upper"
        case "case3_s02_s02_u1_mu5_mu1":
            hull_side = "upper"
        case "case4_s02_s98_u0_mu1_mu5":
            hull_side = "lower"
        case "case5_s98_s02_u1_mu1_mu5":
            hull_side = "upper"
        case "case6_s02_s02_u1_mu1_mu5":
            hull_side = "upper"
        case "case7_s98_s98_u0_mu1_mu5":
            hull_side = "lower"
        case "case8_s98_s98_u0_mu5_mu1":
            hull_side = "lower"
        case "case9_s05_s05_u1_mu1_mu1":
            hull_side = "upper"
        case "case10_s05_s05_u0_mu1_mu1":
            hull_side = "lower"

    # if not run_case.startswith("case10"):
    #     return None

    model_fluxhc1 = FluxHC1(**model_params)
    model_fluxhc2 = ConvexHullFluxHC(**model_params, convex_hull_side=hull_side)
    model_diffhc1 = DiffusionHC(**model_params, fixed_diffusion_coeff=1e-5)
    model_diffhc2 = DiffusionHC(**model_params, fixed_diffusion_coeff=1e-3)
    model_diffhc3 = DiffusionHC(**model_params, fixed_diffusion_coeff=1)

    convex_hull_fig = model_fluxhc2.plot_convex_hull()
    convex_hull_fig.savefig(case_dir / "convex_hull_fw_HC2.png", bbox_inches="tight")

    for final_time in [5.0, 50.0]:
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
                tol=5e-6,
            )
            if solution_curve_fig is not None:
                solution_curve_fig.savefig(
                    case_dir / f"solution_curve_HC_{i}_T_{final_time}.png",
                    bbox_inches="tight",
                )
                plt.close(solution_curve_fig)
            if residual_curve_fig is not None:
                residual_curve_fig.savefig(
                    case_dir / f"residual_curve_HC_{i}_T_{final_time}.png",
                    bbox_inches="tight",
                )
                plt.close(residual_curve_fig)
        if curvature_fig is not None:
            curvature_fig.savefig(
                case_dir / f"curvature_T_{final_time}.png",
                bbox_inches="tight",
            )
            plt.close(curvature_fig)


# Case 1: s_L = 0.02, s_R = 0.98, u_w = 0.0, mu_w = 5.0, mu_n = 1.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [
                jnp.zeros((NUM_CELLS,)),
                jnp.hstack(
                    [
                        jnp.full((NUM_CELLS // 2,), 0.02),
                        jnp.full((NUM_CELLS // 2,), 0.98),
                    ]
                ),
            ],
            axis=1,
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.0], [0.0, 0.0]]),
        "mu_w": 5.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case1_s02_s98_u0_mu5_mu1")

# Case 2: s_L = 0.98, s_R = 0.02, u_w = 1.0, mu_w = 5.0, mu_n = 1.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [
                jnp.zeros((NUM_CELLS,)),
                jnp.hstack(
                    [
                        jnp.full((NUM_CELLS // 2,), 0.98),
                        jnp.full((NUM_CELLS // 2,), 0.02),
                    ]
                ),
            ],
            axis=1,
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.01], [0.0, 0.0]]),
    }
)
compare_solvers(model_params, "case2_s98_s02_u1_mu5_mu1")

# Case 3: s_L = 0.02 = s_R, u_w = 1.0, mu_w = 5.0, mu_n = 1.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [jnp.zeros((NUM_CELLS,)), jnp.full((NUM_CELLS,), 0.02)], axis=1
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.01], [0.0, 0.0]]),
    }
)
compare_solvers(model_params, "case3_s02_s02_u1_mu5_mu1")

# Case 4: s_L = 0.02, s_R = 0.98, u_w = 0.0, mu_w = 1.0, mu_n = 5.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [
                jnp.zeros((NUM_CELLS,)),
                jnp.hstack(
                    [
                        jnp.full((NUM_CELLS // 2,), 0.02),
                        jnp.full((NUM_CELLS // 2,), 0.98),
                    ]
                ),
            ],
            axis=1,
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.0], [0.0, 0.0]]),
        "mu_w": 1.0,
        "mu_n": 5.0,
    }
)
compare_solvers(model_params, "case4_s02_s98_u0_mu1_mu5")


# Case 5: s_L = 0.98, s_R = 0.02, u_w = 0.0, mu_w = 1.0, mu_n = 5.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [
                jnp.zeros((NUM_CELLS,)),
                jnp.hstack(
                    [
                        jnp.full((NUM_CELLS // 2,), 0.98),
                        jnp.full((NUM_CELLS // 2,), 0.02),
                    ]
                ),
            ],
            axis=1,
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.01], [0.0, 0.0]]),
    }
)
compare_solvers(model_params, "case5_s98_s02_u1_mu1_mu5")

# Case 6: s_L = 0.02 = s_R, u_w = 1.0, mu_w = 1.0, mu_n = 5.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [jnp.zeros((NUM_CELLS,)), jnp.full((NUM_CELLS,), 0.02)], axis=1
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.01], [0.0, 0.0]]),
    }
)
compare_solvers(model_params, "case6_s02_s02_u1_mu1_mu5")


# Case 7: s_L = 0.98 = s_R, u_w = 0.0, mu_w = 1.0, mu_n = 5.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [jnp.zeros((NUM_CELLS,)), jnp.full((NUM_CELLS,), 0.98)], axis=1
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.0], [0.0, 0.0]]),
    }
)
compare_solvers(model_params, "case7_s98_s98_u0_mu1_mu5")

# Case 8: s_L = 0.98 = s_R, u_w = 1.0, mu_w = 5.0, mu_n = 1.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [jnp.zeros((NUM_CELLS,)), jnp.full((NUM_CELLS,), 0.98)], axis=1
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.0], [0.0, 0.0]]),
        "mu_w": 5.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case8_s98_s98_u0_mu5_mu1")

# Case 9: s_L = 0.5 = s_R, u_w = 1.0, mu_w = 1.0, mu_n = 1.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [jnp.zeros((NUM_CELLS,)), jnp.full((NUM_CELLS,), 0.5)], axis=1
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.01], [0.0, 0.0]]),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case9_s05_s05_u1_mu1_mu1")

# Case 9: s_L = 0.5 = s_R, u_w = 0.0, mu_w = 1.0, mu_n = 1.0
model_params.update(
    {
        "initial_conditions": jnp.stack(
            [jnp.zeros((NUM_CELLS,)), jnp.full((NUM_CELLS,), 0.5)], axis=1
        ).flatten(),
        "neumann_bc": jnp.array([[0.01, 0.0], [0.0, 0.0]]),
        "mu_w": 1.0,
        "mu_n": 1.0,
    }
)
compare_solvers(model_params, "case10_s05_s05_u0_mu1_mu1")
