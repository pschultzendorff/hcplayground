import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# Nonlinear diffusivity
def k_nonlinear(u):
    return 1 + u**2


# Exact solution (for reference)
def u_exact(x):
    return x * (1 - x)


# Domain
a, b = 0.0, 1.0
N = 20
x_faces = np.linspace(a, b, N + 1)
x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
dx = x_faces[1] - x_faces[0]

# Source term
f = -2 * np.ones(N)

# Solver parameters
tol = 1e-10
max_iter = 50

# --- TPFA with arithmetic average ---
u_tpfa = np.zeros(N)
for _ in range(max_iter):
    u_old = u_tpfa.copy()
    k_u = k_nonlinear(u_tpfa)
    k_face = np.zeros(N + 1)
    k_face[1:-1] = 0.5 * (k_u[:-1] + k_u[1:])
    k_face[0] = k_u[0]
    k_face[-1] = k_u[-1]

    main_diag = (k_face[:-1] + k_face[1:]) / dx**2
    off_diag = -k_face[1:-1] / dx**2

    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1])
    b_vec = f.copy()

    u_tpfa = spsolve(A, b_vec)
    if np.linalg.norm(u_tpfa - u_old, np.inf) < tol:
        break

# --- "MFEM-like" with harmonic mean ---
u_mfem = np.zeros(N)
for _ in range(max_iter):
    u_old = u_mfem.copy()
    k_u = k_nonlinear(u_mfem)
    k_face = np.zeros(N + 1)
    k_face[1:-1] = dx / (0.5 / k_u[:-1] + 0.5 / k_u[1:])
    k_face[0] = k_u[0]
    k_face[-1] = k_u[-1]

    main_diag = (k_face[:-1] + k_face[1:]) / dx**2
    off_diag = -k_face[1:-1] / dx**2

    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1])
    b_vec = f.copy()

    u_mfem = spsolve(A, b_vec)
    if np.linalg.norm(u_mfem - u_old, np.inf) < tol:
        break

# Plotting
x_plot = np.linspace(a, b, 500)
u_tpfa_plot = np.interp(x_plot, x_centers, u_tpfa)
u_mfem_plot = np.interp(x_plot, x_centers, u_mfem)
u_exact_plot = u_exact(x_plot)

plt.figure(figsize=(8, 5))
plt.plot(x_plot, u_tpfa_plot, label="TPFA (avg k(u))", lw=2)
plt.plot(x_plot, u_mfem_plot, label="MFEM-like (harmonic k(u))", lw=2)
plt.plot(x_plot, u_exact_plot, label="Reference: $x(1-x)$", linestyle="--")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.title("Nonlinear Diffusion: TPFA vs. MFEM-like")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(x_plot, u_tpfa_plot, label="TPFA (avg $k(u)$)", lw=2)
plt.plot(x_plot, u_mfem_plot, label="MFEM-like (harmonic $k(u)$)", lw=2)
plt.plot(
    x_plot,
    u_exact_plot,
    label="Reference: $u_{exact}(x) = x(1-x)$",
    linestyle="--",
    color="gray",
)

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Comparison of TPFA and MFEM-like Schemes (Nonlinear Diffusion)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
