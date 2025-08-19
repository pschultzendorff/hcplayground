"""Implicit finite volume model for two-phase flow in porous media.


An example geometry with the domain discretized into two cells looks like this:

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
transmissibilities are fixed beforehand) and phase-potential upwinding.

"""

from typing import Optional

import jax
import jax.numpy as jnp


class TPFModel:
    def __init__(self, **kwargs) -> None:
        # General model and solver parameters.
        self.num_cells = kwargs["num_cells"]
        self.domain_size = kwargs.get("domain_size", 1.0)
        self.permeabilities = kwargs["permeabilities"]
        self.porosities = kwargs["porosities"]
        self.source_terms = kwargs["source_terms"]
        self.bc = kwargs["bc"]
        self.neumann_bc = kwargs["neumann_bc"]
        self.dirichlet_bc = kwargs["dirichlet_bc"]
        self.initial_conditions = kwargs["initial_conditions"]
        self.mu_w = kwargs["mu_w"]
        self.mu_n = kwargs["mu_n"]

        # Constitutive laws and parameters.
        self.cp_model = kwargs.get("pc_model", "Brooks-Corey")
        self.rp_model = kwargs.get("rp_model", "Brooks-Corey")
        self.nb = kwargs.get("nb", 2)
        self.p_e = kwargs.get("p_e", 5.0)
        self.n1 = kwargs.get("n1", 2)
        self.n2 = kwargs.get("n2", 1 + 2 / self.nb)
        self.n3 = kwargs.get("n3", 1)

        # Placeholder for the linear system (Jacobian, residual).
        self.linear_system: tuple = (None, None)

        self.initialize_transmissibilities()

    def initialize_transmissibilities(self):
        """Initialize cell face transmissibilities with two-point flux approximation
        (TPFA).

        """
        # Harmonic average of half-transmissibilities gives the cell faces
        # transmissibilities.
        half_transmissibilities = (
            self.domain_size / (self.num_cells * 2) * self.permeabilities
        )
        assert (half_transmissibilities > 0).all(), "Permeabilities must be positive."
        half_transmissibilities_inverse = jnp.pow(half_transmissibilities, -1)
        self.transmissibilities = jnp.pow(
            half_transmissibilities_inverse[1:] + half_transmissibilities_inverse[:-1],
            -1,
        )

        # Double transmissibilities at Dirichlet boundaries.
        if self.bc[0] == "dirichlet":
            self.transmissibilities = jnp.insert(
                self.transmissibilities, 0, self.transmissibilities[0] * 2
            )
        if self.bc[1] == "dirichlet":
            self.transmissibilities = jnp.append(
                self.transmissibilities, self.transmissibilities[-1] * 2
            )

        # Null transmissibilities at Neumann boundaries.
        if self.bc[0] == "neumann":
            self.transmissibilities = jnp.insert(self.transmissibilities, 0, 0.0)
        if self.bc[1] == "neumann":
            self.transmissibilities = jnp.append(self.transmissibilities, 0.0)

    # We ignore the possiblity of the user passing invalid models for the constitutive
    # laws.
    def pc(self, s: jnp.ndarray, cp_model: Optional[str] = None) -> jnp.ndarray:  # type: ignore
        """Capillary pressure function. Brooks-Corey model.

        Limit to above to avoid problems with Newton when :math:`s_w = 0`.

        """
        if cp_model is None:
            cp_model = self.cp_model
        if cp_model == "Brooks-Corey":
            return jnp.minimum(
                jnp.nan_to_num(
                    self.p_e * s ** (-1 / self.nb),
                    nan=0.0,
                    posinf=self.p_e * 10,
                    neginf=0.0,
                ),
                self.p_e * 10,
            )
        elif cp_model == "linear":
            return self.p_e * (2 - s)
        elif cp_model == "zero":
            return jnp.zeros_like(s)

    def mobility_w(self, s: jnp.ndarray, rp_model: Optional[str] = None) -> jnp.ndarray:
        """Mobility of the wetting phase. Brooks-Corey model."""
        if rp_model is None:
            rp_model = self.rp_model
        if rp_model == "Brooks-Corey":
            k_w = jnp.where(s >= 0, s ** (self.n1 + self.n2 * self.n3), 0.0)
        elif rp_model == "Corey":
            k_w = jnp.where(s >= 0, s**2, 0.0)
        elif rp_model == "linear":
            k_w = s
        return k_w / self.mu_w  # type: ignore

    def mobility_n(self, s: jnp.ndarray, rp_model: Optional[str] = None) -> jnp.ndarray:
        """Mobility of the nonwetting phase. Brooks-Corey model."""
        if rp_model is None:
            rp_model = self.rp_model
        if rp_model == "Brooks-Corey":
            k_n = jnp.where(
                s <= 1, (1 - s) ** self.n1 * (1 - s**self.n2) ** self.n3, 0.0
            )
        elif rp_model == "Corey":
            k_n = jnp.where(s <= 1, (1 - s) ** 2, 0.0)
        elif rp_model == "linear":
            k_n = 1 - s
        return k_n / self.mu_n  # type: ignore

    def compute_face_fluxes(self, p, s):
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
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        p_c = self.pc(s)

        # Calculate fluxes at each face. After adding Dirichlet bc, the indices work as
        # follows. Face i has cell i on the left and cell i + 1 on the right. The negative
        # pressure gradient across cell i is therefore given by p[i] - p[i + 1].

        # TPFA fluxes at the faces.
        p_n_gradient = p[:-1] - p[1:]  # Negative pressure gradient across cells.
        p_c_gradient = (
            p_c[:-1] - p_c[1:]
        )  # Negative capillary pressure gradient across cells.

        # Phase potential upwinding of the mobilities. The mobilities for Neumann
        # boundaries are upwinded as well. This is not a problem, because the
        # transmissibilities are set to 0 at the Neumann boundary.
        m_w = jnp.where(
            p_n_gradient > 0, self.mobility_w(s[:-1]), self.mobility_w(s[1:])
        )
        m_n = jnp.where(
            p_n_gradient - p_c_gradient >= 0,
            self.mobility_n(s[:-1]),
            self.mobility_n(s[1:]),
        )

        # Handle NaN values in mobilities.
        m_n = jnp.nan_to_num(m_n, nan=0.0)
        m_w = jnp.nan_to_num(m_w, nan=0.0)

        m_t = m_n + m_w

        F_t = self.transmissibilities * (m_t * p_n_gradient - m_w * p_c_gradient)
        F_w = (
            F_t * (m_w / m_t) - self.transmissibilities * m_w * m_n / m_t * p_c_gradient
        )

        # Add Neumann boundary conditions.
        if self.bc[0] == "neumann":
            F_t = F_t.at[0].set(self.neumann_bc[0, 0])
            F_w = F_w.at[0].set(self.neumann_bc[0, 1])
        if self.bc[1] == "neumann":
            F_t = F_t.at[-1].set(self.neumann_bc[1, 0])
            F_w = F_w.at[-1].set(self.neumann_bc[1, 1])

        return F_t, F_w

    def residual(self, x, dt, x_prev=None):
        """Compute the residual of the system at (x, beta)."""
        p = x[::2]
        s = x[1::2]

        if x_prev is None:
            s_prev = self.initial_conditions[1::2]
        else:
            s_prev = x_prev[1::2]

        # Compute fluxes.
        F_t, F_w = self.compute_face_fluxes(p, s)

        # Residuals for flow and transport equations.
        r_f = F_t[1:] - F_t[:-1] - self.source_terms[:, 0]
        r_t = (
            self.porosities * (s - s_prev) / dt
            + F_w[1:]
            - F_w[:-1]
            - self.source_terms[:, 1]
        )

        r = jnp.concatenate([r_f, r_t])
        return r

    def jacobian(self, x, dt, x_prev=None):
        """Compute the Jacobian of the system at (x, beta)."""
        return jax.jacrev(self.residual, argnums=0)(x, dt, x_prev=x_prev)
