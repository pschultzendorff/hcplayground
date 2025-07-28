import pathlib
import subprocess
import sys

import jax.numpy as jnp
import viztracer

parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from hc import DiffusionHC  # noqa: E402
from plotting import solve_and_plot  # noqa: E402

results_dir = pathlib.Path(__file__).parent / "benchmark"
results_dir.mkdir(exist_ok=True)

NUM_CELLS = 30

model_params = {
    "num_cells": NUM_CELLS,
    "transmissibilities": jnp.full((NUM_CELLS + 1,), 0.1),
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


tracer = viztracer.VizTracer(min_duration=1e5)
tracer.start()

model = DiffusionHC(**model_params, fixed_diffusion_coeff=1.0)
final_time = 10.0
solve_and_plot(
    model,
    final_time,
    "blue",
    f"{model.__class__.__name__}",
    hc_max_iter=100,
    max_iter=10,
    appleyard=True,
)

tracer.stop()
tracer.save(str(results_dir / "benchmark.json"))

subprocess.run(["vizviewer", str(results_dir / "benchmark.json")])
