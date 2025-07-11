"""Curvature for a 1D two-cell two-phase flow problem solved with HC.


The example geometry looks like this:

          |-------------------|-------------------|
          |                   |                   |
Neu bc    F0                  F1                  F2      Dir bc
       ------>    p0,s0    ------>    p1,s1    ------>    p2,s2
          |                   |                   |
          |-------------------|-------------------|
          e0        c0        e1        c1        e2

The two-phase flow problem, formulated in the fractional flow formulation, is solved for
nonwetting pressure and wetting saturation.

The problem is discretized with implicit Euler in time and cell-centered finite volumes
in space. The face fluxes are evaluated with two-point flux approximation (where the
transmissibilities are fixed beforehand) and phase-potential upwinding. The resulting
system of nonlinear equations at each time step is solved with a homotopy continuation
method, where the corrector step uses Newton's method.


"""

import numpy as np
import scipy as sp

NUM_CELLS = 2

TRANSMISSIBILITIES = np.array(
    [0.0, 1.0, 2.0]
)  # Transmissibilities for faces e0, e1, e2.
# Transmissibility at the Dirichlet boundary has to be doubled.
# Transmissibility at the Neumann boundary hast to be set to 0.

POROSITIES = np.array([0.5, 0.5])  # Porosities for cells c0, c1.
SOURCE_TERMS = np.array(
    [[0.0, 0.0], [0.0, 0.0]]
)  # Source terms for cells c0, c1. Total and wetting source.

NEU_BC = np.array(
    [
        [1.0, 0.5],
        [0.0, 0.0],
    ]
)  # Neumann boundary conditions for faces e0, e1. Total flux and wetting flux.
DIR_BC = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.5],
    ]
)  # Dirichlet boundary conditions for faces e1, e2. Pressure and saturation.

INITIAL_CONDITIONS = np.array(
    [0.0, 0.5, 0.0, 0.5]
)  # Initial conditions for [p0, s0, p1, s1].


MU_W = 1.0  # Viscosity of the wetting phase.
MU_N = 1.0  # Viscosity of the nonwetting phase.

beta = 1.0  # Homotopy parameter, 0 for simpler system, 1 for actual system. For simplicity, we use a global variable.


def pc(s):
    """Capillary pressure function."""
    return 0.1 * (1 - s) / s  # Example function

def pc_prime(s):
    """Derivative of the capillary pressure function."""
    return -0.1 * (1 - s) / s**2  # Derivative of the example function 

def mobility_w(s):
    """Mobility of the wetting phase."""
    k_w = beta * s + (1 - beta) * s**2
    return k_w / MU_W


def mobility_n(s):
    """Mobility of the nonwetting phase."""
    k_n = beta * (1 - s) + (1 - beta) * (1 - s) ** 2
    return k_n / MU_N


def mobility_w_prime(s):
    """Derivative of the wetting phase mobility."""
    return (beta + 2 * (1 - beta) * s) / MU_W


def mobility_n_prime(s):
    """Derivative of the nonwetting phase mobility."""
    return (-beta + 2 * (1 - beta) * (1 - s)) / MU_N


def compute_face_fluxes(p, s):
    """Compute total and wetting phase fluxes at cell faces.

    Note: The code is written general to handle different boundary conditions and source
    terms.

    Args:
        p: Nonwetting pressures at cell centers [p0, p1].
        s: Wetting saturations at cell centers [s0, s1].

    Returns:
        F_t: Total fluxes at faces [F0, F1, F2].
        F_w: Wetting phase fluxes at faces [F0, F1, F2].

    """
    # Add Dirichlet boundary conditions.
    p = np.concat(DIR_BC[0, :1], p, DIR_BC[1, :1])
    s = np.concat(DIR_BC[0, 1:], s, DIR_BC[1, 1:])

    p_c = pc(s)

    # Calculate fluxes at each face. After adding Dirichlet bc, the indices work as
    # follows. Face i has cell i on the left and cell i + 1 on the right. The negative
    # pressure gradient across cell i is therefore given by p[i] - p[i + 1].

    # Phase potential upwinding of the mobilities. The mobilities for Neumann
    # boundaries are upwinded as well. This is not a problem, because the
    # transmissibilities are set to 0 at the Neumann boundary.
    m_w = np.zeros(NUM_CELLS + 1)
    m_n = np.zeros(NUM_CELLS + 1)

    for i in range(NUM_CELLS + 1):
        m_n[i] = mobility_w(s[i]) if p[i] >= p[i + 1] else mobility_w(s[i + 1])
        m_w[i] = (
            mobility_n(s[i])
            if p[i] - p_c[i] >= p[i + 1] - p_c[i + 1]
            else mobility_n(s[i + 1])
        )

    m_t = m_n + m_w

    # TPFA fluxes at the faces.
    p_n_gradient = p[:-1] - p[1:]  # Negative pressure gradient across cells.
    p_c_gradient = p_c[:-1] - p_c[:-1]  # Negative capillary pressure gradient across cells.

    F_t = TRANSMISSIBILITIES * m_t * p
    F_t = TRANSMISSIBILITIES * (
        m_t * p_n_gradient - m_w * p_c_gradient
    )
    F_w = F_t * (m_w / m_t) - TRANSMISSIBILITIES[
        i
    ] * m_w * m_n / m_t * p_c_gradient

    # Add Neumann boundary conditions.
    F_t[0] += NEU_BC[0, 0]
    F_w[0] += NEU_BC[0, 1]
    F_t[-1] += NEU_BC[1, 0]
    F_w[-1] += NEU_BC[1, 1]

    return F_t, F_w


def residual(x, x_prev=None, dt=1.0):
    """Compute the residual of the system at (x, beta)."""
    p = x[[0, 2]]
    s = x[[1, 3]]

    if x_prev is None:
        s_prev = INITIAL_CONDITIONS[[1, 3]]
    else:
        s_prev = x_prev[[1, 3]]

    # Compute fluxes.
    F_t, F_w = compute_face_fluxes(p, s)

    # Residuals for flow and transport equations.
    r_f = F_t[1:] - F_t[:-1] - SOURCE_TERMS[:, 0]
    r_t = POROSITIES * (s - s_prev) / dt + F_w[1:] - F_w[:-1] - SOURCE_TERMS[:, 1]

    r = np.concatenate([r_f, r_t])
    return r


def jacobian(x, dt=1.0):
    """Compute the Jacobian of the system at (x, beta)."""
    p = x[[0, 2]]
    s = x[[1, 3]]

    # Add Dirichlet boundary conditions.
    p = np.concat(DIR_BC[0, :1], p, DIR_BC[1, :1])
    s = np.concat(DIR_BC[0, 1:], s, DIR_BC[1, 1:])

    p_c = pc(s)

    # Upwind the mobilities and their derivatives. The mobilities for Neumann boundaries
    # are upwinded as well. This is not a problem, because the transmissibilities are
    # set to 0 at the Neumann boundary.
    m_w = np.zeros(NUM_CELLS + 1)
    m_n = np.zeros(NUM_CELLS + 1)
    m_w_prime = np.zeros(NUM_CELLS + 1)
    m_n_prime = np.zeros(NUM_CELLS + 1)

    for i in range(NUM_CELLS + 1):
        m_n[i] = mobility_w(s[i]) if p[i] >= p[i + 1] else mobility_w(s[i + 1])
        m_w[i] = (
            mobility_n(s[i])
            if p[i] - p_c[i] >= p[i + 1] - p_c[i + 1]
            else mobility_n(s[i + 1])
        )

        m_n_prime[i] = (
            mobility_w_prime(s[i]) if p[i] >= p[i + 1] else mobility_w_prime(s[i + 1])
        )
        m_w_prime[i] = (
            mobility_n_prime(s[i])
            if p[i] - p_c[i] >= p[i + 1] - p_c[i + 1]
            else mobility_n_prime(s[i + 1])
        )

    m_t = m_n + m_w
    m_t_prime = m_n_prime + m_w_prime

    p_n_gradient = p[:-1] - p[1:]  # Negative pressure gradient across cells.
    p_c_gradient = p_c[:-1] - p_c[:-1]  # Negative capillary pressure gradient across cells.

    J = np.zeros((NUM_CELLS * 2, NUM_CELLS * 2))

    # Flow equation contributions to the Jacobian.
    dF_t_dp = TRANSMISSIBILITIES * m_t
    dF_w_dp = TRANSMISSIBILITIES * m_w
    dF_t_ds = POROSITIES / dt + TRANSMISSIBILITIES * m_t_prime
    dF_t_ds_next = 
    for i in range(NUM_CELLS + 1):
        # Derivative of the total flux with respect to saturation.
        dF_t_ds = TRANSMISSIBILITIES[i] * (
            mobility_w(s[i]) / MU_W + mobility_n(s[i]) / MU_N
        )
        dF_t_ds_next = TRANSMISSIBILITIES[i + 1] * (
            mobility_w(s[i + 1]) / MU_W + mobility_n(s[i + 1]) / MU_N
        )

        # Fill the Jacobian for the flow equation.
        J[i, i] = -dF_t_dp - dF_t_ds
        J[i, i + 1] = dF_t_dp_next
        J[i + NUM_CELLS, i] = -dF_t_ds
        J[i + NUM_CELLS, i + 1] = dF_t_ds

    # Transport equation contributions to the Jacobian.
    for i in range(NUM_CELLS):
        # Derivative of the wetting phase flux with respect to saturation.
        dF_w_ds = TRANSMISSIBILITIES[i] * (
            mobility_w_prime(s[i]) / MU_W + mobility_n_prime(s[i]) / MU_N
        )
        dF_w_ds_next = TRANSMISSIBILITIES[i + 1] * (
            mobility_w_prime(s[i + 1]) / MU_W + mobility_n_prime(s[i + 1]) / MU_N
        )

        # Fill the Jacobian for the transport equation.
        J[i + NUM_CELLS, i] = POROSITIES[i] / dt - dF_w_ds
        J[i + NUM_CELLS, i + 1] = -dF_w_ds_next
        J[i + NUM_CELLS + NUM_CELLS, i] = -POROSITIES[i] / dt
        J[i + NUM_CELLS + NUM_CELLS, i + 1] = dF_w_ds_next

    return J


def newton(x_init, max_iter=50, tol=1e-6):
    """Solve the system using Newton's method."""
    x = x_init.copy()

    for i in range(max_iter):
        r = residual(x)
        if np.linalg.norm(r) < tol:
            break

        J = jacobian(x)
        dx = np.linalg.solve(J, -r)
        x += dx

    return x


def homotopy_continuation(x_init, n_steps=10, max_newton_iter=20, tol=1e-6):
    """Solve the system using homotopy continuation with Newton's method."""
    x = x_init.copy()

    for i in range(n_steps):
        beta = (i + 1) / n_steps

        # Newton's method for the corrector step
        for j in range(max_newton_iter):
            r = residual(x, beta)
            if np.linalg.norm(r) < tol:
                break

            J = jacobian(x, beta)
            dx = np.linalg.solve(J, -r)
            x += dx

            # Add damping if needed
            if j > max_newton_iter / 2:
                x += 0.5 * dx

    return x


def solve_time_step(x_prev, dt=1.0):
    """Solve one time step of the two-phase flow problem."""
    return homotopy_continuation(x_prev)


def solve(initial_conditions, final_time, n_time_steps):
    """Solve the two-phase flow problem over a given time period."""
    dt = final_time / n_time_steps
    solutions = [initial_conditions]

    for i in range(n_time_steps):
        x_prev = solutions[-1]
        x_new = solve_time_step(x_prev, dt)
        solutions.append(x_new)

    return solutions


def h_beta_deriv(x, x_prev=None):
    """Compute the derivative of the homotopy problem with respect to beta."""
    global beta
    beta_save = beta

    beta = 1.0
    r_g = residual(x, x_prev=x_prev)
    beta = 0.0
    r_f = residual(x, x_prev=x_prev)

    beta = beta_save
    return r_g - r_f


def curve_tangent(x, x_prev=None, jac=None):
    """Compute the tangent vector of the homotopy curve at (x, beta)."""
    b = h_beta_deriv(x, x_prev=x_prev)
    A = jacobian(x, x_prev=x_prev) if jac is None else jac
    tangent = np.linalg.solve(A, b)
    return tangent


def hessian(x, x_prev=None):
    """Compute the Hessian of the homotopy problem at (x, beta)."""
    n = len(x)

    def hessian_tensor(v1, v2):
        """Compute the Hessian tensor of the homotopy problem."""
        hessian_matrix = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            hessian_matrix[:, i] = curve_tangent(x + e_i, x_prev=x_prev)

        return hessian_matrix

    return hessian_tensor


def curve_curvature(x, x_prev=None, jac=None):
    if jac is None:
        jac = jacobian(x, x_prev=x_prev)
    tangent = curve_tangent(x, x_prev=x_prev, jac=jac)
    w2 = hessian(x, x_prev=x_prev)(tangent, tangent)
    curvature_vector = np.linalg.solve(jac, -w2)
    return np.linalg.norm(curvature_vector)
