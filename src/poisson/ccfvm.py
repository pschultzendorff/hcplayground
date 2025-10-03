import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

dirname = pathlib.Path(__file__).parent


def exact_solution(x):
    return -(x**2) + x


# region Grid and Problem Setup


def generate_grid(domain_start, domain_end, num_cells):
    """Generates uniform grid cell faces and centers."""
    cell_faces = np.linspace(domain_start, domain_end, num_cells + 1)
    cell_centers = 0.5 * (cell_faces[:-1] + cell_faces[1:])
    return cell_faces, cell_centers


def setup_problem(domain_start, domain_end, num_cells, heterogeneous=True):
    """Defines grid, source term, and heterogeneous permeability."""
    cell_faces, cell_centers = generate_grid(domain_start, domain_end, num_cells)
    cell_width = (domain_end - domain_start) / num_cells

    # Define piecewise constant source term
    source_term = np.full(num_cells, 2.0)
    # source_term[1::2] = -10.0
    # # Make the source term non-symmetric
    # source_term[2] = -10.0

    if heterogeneous:
        # Heterogeneous permeability
        permeability = np.random.uniform(0.1, 10.0, size=num_cells)
    else:
        permeability = np.full((num_cells,), 1)

    return cell_faces, cell_centers, cell_width, source_term, permeability


# endregion


# region Two-Point Flux Discretization (TPFA)


def harmonic_mean(k1, k2):
    """Computes the harmonic mean of two values."""
    return 2 * k1 * k2 / (k1 + k2)


def assemble_system(cell_faces, cell_width, source, permeability):
    """
    Assembles the FVM system matrix and RHS using two-point flux approximation.
    Returns sparse matrix A, RHS vector b, and face transmissibilities.
    """
    num_cells = len(source)
    num_faces = num_cells + 1

    transmissibility = np.zeros(num_faces)

    # Interior transmissibilities (harmonic average)
    for i in range(1, num_cells):
        k_harm = harmonic_mean(permeability[i - 1], permeability[i])
        transmissibility[i] = k_harm / cell_width

    # Boundary transmissibilities (half-cell width)
    transmissibility[0] = permeability[0] / (0.5 * cell_width)
    transmissibility[-1] = permeability[-1] / (0.5 * cell_width)

    # Assemble tridiagonal matrix A
    main_diag = transmissibility[:-1] + transmissibility[1:]
    lower_diag = -transmissibility[1:-1]
    upper_diag = -transmissibility[1:-1]

    A = diags(
        diagonals=[lower_diag, main_diag, upper_diag],
        offsets=[-1, 0, 1],  # type: ignore
        shape=(num_cells, num_cells),
        format="csr",
    )

    b = source * cell_width
    return A, b, transmissibility


# endregion


# region Flux Computation


def compute_face_fluxes(cell_values, transmissibility):
    """
    Computes fluxes at cell faces using the two-point flux approximation.
    Assumes Dirichlet zero boundary conditions.
    """
    num_cells = len(cell_values)
    num_faces = num_cells + 1

    # Extend solution vector with ghost values at boundaries (Dirichlet BCs: u=0)
    u_ext = np.zeros(num_cells + 2)
    u_ext[1:-1] = cell_values

    # Compute fluxes at each face
    flux = np.zeros(num_faces)
    for i in range(num_faces):
        flux[i] = -transmissibility[i] * (u_ext[i + 1] - u_ext[i])

    return flux


# endregion


# region Flux Function σ_h(x)


def build_flux_function(cell_faces, face_fluxes):
    """
    Constructs a piecewise linear flux function σ_h(x), continuous on the domain.
    """
    dx = cell_faces[1] - cell_faces[0]

    def sigma(x):
        if x <= cell_faces[0]:
            return face_fluxes[0]
        elif x >= cell_faces[-1]:
            return face_fluxes[-1]

        i = int((x - cell_faces[0]) // dx)
        xL = cell_faces[i]
        slope = (face_fluxes[i + 1] - face_fluxes[i]) / dx
        intercept = face_fluxes[i] - slope * xL
        return slope * x + intercept

    return sigma


# endregion

# region Quadratic Potential Reconstruction (Piecewise)


def reconstruct_potential_quadratic(cell_faces, fluxes, diffusivity, uh):
    """
    Reconstructs a piecewise quadratic potential on each cell (i).
    Uses:
        - fluxes at cell edges (i, i+1)
        - local cell-average potential uh[i]
        - cell-wise constant diffusivity k[i]

    The reconstructed potential p_i(x) satisfies:
        - -k_i * dp/dx(x_i)   = fluxes[i]
        - -k_i * dp/dx(x_{i+1}) = fluxes[i+1]
        - average of p_i(x) over cell = uh[i]
    """
    dx = cell_faces[1] - cell_faces[0]
    coeffs = []

    for i in range(len(uh)):
        xL = cell_faces[i]
        xR = cell_faces[i + 1]
        k = diffusivity[i]
        sigma_L = fluxes[i]
        sigma_R = fluxes[i + 1]

        # dp/dx(xL) = -sigma_L / k, dp/dx(xR) = -sigma_R / k
        dpdx_L = -sigma_L / k
        dpdx_R = -sigma_R / k

        # Fit linear gradient to match values at xL and xR
        slope = (dpdx_R - dpdx_L) / dx
        intercept = dpdx_L - slope * xL

        # Integrate gradient to get potential: p(x) = a x^2 + b x + c
        a = 0.5 * slope
        b = intercept

        # Integrate potential over cell to match average uh[i]
        def integral_p(x):
            return a * x**3 / 3 + b * x**2 / 2

        avg_integral = (integral_p(xR) - integral_p(xL)) / dx
        c = uh[i] - avg_integral

        coeffs.append((a, b, c))

    return coeffs


def eval_quadratic(coeffs, faces, x):
    """Evaluates piecewise quadratic function at point x."""
    if x < faces[0] or x > faces[-1]:
        return 0.0
    dx = faces[1] - faces[0]
    i = min(int((x - faces[0]) // dx), len(coeffs) - 1)
    a, b, c = coeffs[i]
    return a * x**2 + b * x + c


# endregion


# region Shifted Potential (Elementwise Zero at Endpoints)


def build_shifted_potential(coeffs, faces):
    """Shifts each quadratic so that it vanishes at cell endpoints (average).

    Note: This only works for if the reconstructed potential attains the same values at
    each cells endpoints.

    """
    shifted_coeffs = []

    for i, (a, b, c) in enumerate(coeffs):
        xL, xR = faces[i], faces[i + 1]
        pL = a * xL**2 + b * xL + c
        pR = a * xR**2 + b * xR + c
        shift = 0.5 * (pL + pR)
        shifted_coeffs.append((a, b, c - shift))

    return shifted_coeffs


# endregion


# region Oswald Interpolation (Global Conforming Potential)


def oswald_interpolate(coeffs, cell_faces, cell_centers):
    """Builds a globally conforming quadratic potential via Oswald interpolation."""
    num_cells = len(coeffs)
    num_nodes = num_cells + 1
    node_vals = np.zeros(num_nodes)
    node_counts = np.zeros(num_nodes)

    # Average values at cell faces (nodes)
    for i, (a, b, c) in enumerate(coeffs):
        xL = cell_faces[i]
        xR = cell_faces[i + 1]
        pL = a * xL**2 + b * xL + c
        pR = a * xR**2 + b * xR + c
        node_vals[i] += pL
        node_vals[i + 1] += pR
        node_counts[i] += 1
        node_counts[i + 1] += 1

    node_vals /= node_counts
    node_vals[0] = 0.0  # Dirichlet BC
    node_vals[-1] = 0.0

    # Midpoints (evaluate directly)
    mid_vals = np.array(
        [a * xm**2 + b * xm + c for (a, b, c), xm in zip(coeffs, cell_centers)]
    )

    # Fit P2 polynomial in each cell from (node_left, midpoint, node_right)
    conforming_coeffs = []

    for i in range(num_cells):
        x_pts = np.array([cell_faces[i], cell_centers[i], cell_faces[i + 1]])
        y_pts = np.array([node_vals[i], mid_vals[i], node_vals[i + 1]])
        V = np.vstack([x_pts**2, x_pts, np.ones(3)]).T
        coeff = np.linalg.solve(V, y_pts)
        conforming_coeffs.append(tuple(coeff))

    return conforming_coeffs


# endregion


# region Energy Norm Computation


def compute_energy_norm(p1_coeffs, p2_coeffs, faces):
    """Computes H1-seminorm of the difference between two potentials."""
    error_sq = 0.0

    for i in range(len(p1_coeffs)):
        a1, b1, _ = p1_coeffs[i]
        a2, b2, _ = p2_coeffs[i]
        xL, xR = faces[i], faces[i + 1]

        def squared_grad_diff(x):
            return (2 * a1 * x + b1 - (2 * a2 * x + b2)) ** 2

        integral, _ = quad(squared_grad_diff, xL, xR)
        error_sq += integral

    return np.sqrt(error_sq)


# endregion


# region Plotting


def plot_results(
    faces, centers, u_h, sigma_h_func, quad_coeffs, shifted_coeffs, oswald_coeffs
):
    """Plots FVM solution and all reconstructed potentials with cellwise piecewise quadratics."""

    def plot_piecewise_quadratic(ax, coeffs, faces, label):
        """Plot piecewise quadratic functions cellwise to show discontinuities."""
        for i in range(len(coeffs)):
            a, b, c = coeffs[i]
            x_vals = np.linspace(faces[i], faces[i + 1], 50)
            y_vals = a * x_vals**2 + b * x_vals + c
            ax.plot(x_vals, y_vals, color="C0")
        ax.set_title(label)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$p(x)$")
        ax.grid(True)

    x_fine = np.linspace(faces[0], faces[-1], 500)
    # u_exact_fine = exact_solution(x_fine)
    sigma_vals = np.array([sigma_h_func(x) for x in x_fine])

    fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(12, 8))

    # Plot piecewise constant solution
    for i in range(len(u_h)):
        ax1.plot(
            [faces[i], faces[i + 1]],
            [u_h[i], u_h[i]],
            color="C2",
            label="$u_h$" if i == 0 else None,
        )
    ax1.set_title("Piecewise Constant FVM Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.grid(True)
    ax1.legend()

    # Flux function
    ax2.plot(x_fine, sigma_vals, label=r"$\sigma_h(x)$")
    ax2.set_title("Equilibrated Flux Function")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"$\sigma(x)$")
    ax2.grid(True)
    ax2.legend()

    # Piecewise Quadratic Potential (cellwise)
    plot_piecewise_quadratic(ax3, quad_coeffs, faces, "Piecewise Quadratic Potential")

    # Shifted Potential (cellwise)
    plot_piecewise_quadratic(
        ax4, shifted_coeffs, faces, "Shifted (Conforming) Reconstruction"
    )

    # Oswald Interpolated Potential (cellwise)
    plot_piecewise_quadratic(
        ax5, oswald_coeffs, faces, "Oswald Interpolated Reconstruction"
    )

    fig.tight_layout()
    return fig


# endregion

# region Setup and solve problem

domain_start, domain_end = 0.0, 1.0
num_cells = 1

# Setup grids and data for heterogeneous and homogeneous diffusivity
faces, centers, cell_width, source_vals, diff_hetero = setup_problem(
    domain_start, domain_end, num_cells
)
_, _, _, _, diff_homo = setup_problem(
    domain_start, domain_end, num_cells, heterogeneous=False
)

# Assemble and solve linear systems
A_hetero, rhs_hetero, transmissibility_hetero = assemble_system(
    faces, cell_width, source_vals, diff_hetero
)
solution_hetero = spsolve(A_hetero, rhs_hetero)

A_homo, rhs_homo, transmissibility_homo = assemble_system(
    faces, cell_width, source_vals, diff_homo
)
solution_homo = spsolve(A_homo, rhs_homo)

# endregion


# region Postprocessing, error computation and visualization


def process_solution(
    faces, centers, diffusivity, transmissibility, solution, label="Solution"
):
    """
    Postprocess solution: compute fluxes, reconstruct potentials, calculate energy errors, and plot results.
    """
    # Compute fluxes at cell faces using transmissibility matrix and solution
    fluxes = compute_face_fluxes(solution, transmissibility)

    # Construct equilibrated flux function σ_h(x)
    flux_function = build_flux_function(faces, fluxes)

    # Reconstruct quadratic potentials (piecewise per cell)
    quad_coeffs = reconstruct_potential_quadratic(faces, fluxes, diffusivity, solution)

    # Shifted quadratic potentials (zero at cell endpoints)
    shifted_coeffs = build_shifted_potential(quad_coeffs, faces)

    # Conforming potential reconstruction via Oswald interpolation
    oswald_coeffs = oswald_interpolate(quad_coeffs, faces, centers)

    # Compute energy norm errors between postprocessed potentials
    err_shift = compute_energy_norm(quad_coeffs, shifted_coeffs, faces)
    err_oswald = compute_energy_norm(quad_coeffs, oswald_coeffs, faces)

    print(f"{label} - Energy norm ||u_quad - u_shift||_E = {err_shift:.6e}")
    print(f"{label} - Energy norm ||u_quad - u_oswald||_E = {err_oswald:.6e}")

    # Visualization
    fig = plot_results(
        faces,
        centers,
        solution,
        flux_function,
        quad_coeffs,
        shifted_coeffs,
        oswald_coeffs,
    )
    fig.savefig(dirname / f"{label.replace(' ', '_').lower()}_results.png")


# Process and plot heterogeneous diffusivity results
process_solution(
    faces,
    centers,
    diff_hetero,
    transmissibility_hetero,
    solution_hetero,
    label="Heterogeneous",
)

# Process and plot homogeneous diffusivity results
process_solution(
    faces,
    centers,
    diff_homo,
    transmissibility_homo,
    solution_homo,
    label="Homogeneous",
)
process_solution(
    faces,
    centers,
    diff_homo,
    transmissibility_hetero,
    solution_homo,
    label="Homogeneous solution - Heterogeneous postprocessing",
)

# endregion
