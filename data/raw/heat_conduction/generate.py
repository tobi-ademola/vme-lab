import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from fenics import (
    UnitSquareMesh, FunctionSpace, DirichletBC, Constant, near,
    TrialFunction, TestFunction, Function, dot, grad, dx, solve,
    File,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES   = 500
MESH_N      = 32          # 32×32 elements → 33×33 vertex grid
RANDOM_SEED = 42

# Physical parameter bounds: [k (W/m·K), Q (W/m³), T_left (K), T_right (K)]
PARAM_NAMES       = ["k", "Q", "T_left", "T_right"]
PARAM_BOUNDS_LOW  = np.array([  1.0,    0.0, 300.0, 300.0])
PARAM_BOUNDS_HIGH = np.array([ 50.0, 1000.0, 600.0, 600.0])

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR      = os.path.join(SCRIPT_DIR, "output", "plots")
VTK_DIR       = os.path.join(SCRIPT_DIR, "output", "vtk")
DATASET_PATH  = os.path.join(SCRIPT_DIR, "heat_conduction.npz")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "..", "processed", "heat_conduction")

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

# Build mesh once — reused for every solve
_mesh = UnitSquareMesh(MESH_N, MESH_N)
_V    = FunctionSpace(_mesh, "P", 1)

# Precompute coordinate-based sort index so vertex values can be reshaped
# into a regular (ny+1, nx+1) grid (sorted by y then x)
_coords   = _mesh.coordinates()                          # (n_verts, 2)
_sort_idx = np.lexsort((_coords[:, 0], _coords[:, 1]))  # row-major: x varies fastest


def solve_heat_conduction(
    k_val: float,
    Q_val: float,
    T_left: float,
    T_right: float,
) -> tuple[Function, np.ndarray]:
    """Solve steady-state 2D heat conduction on the unit square.

    PDE  : -∇·(k ∇T) = Q
    BCs  : T = T_left at x=0, T = T_right at x=1
           ∂T/∂n = 0 at y=0 and y=1 (insulated, natural Neumann)

    Returns
    -------
    T_fn  : FEniCS Function (P1) — needed for VTK output
    field : np.ndarray of shape (MESH_N+1, MESH_N+1), dtype float32
    """
    bc_left  = DirichletBC(_V, Constant(T_left),  lambda x, ob: ob and near(x[0], 0.0))
    bc_right = DirichletBC(_V, Constant(T_right), lambda x, ob: ob and near(x[0], 1.0))

    T = TrialFunction(_V)
    v = TestFunction(_V)
    a = Constant(k_val) * dot(grad(T), grad(v)) * dx
    L = Constant(Q_val) * v * dx

    T_fn = Function(_V)
    solve(a == L, T_fn, [bc_left, bc_right])

    # Sort vertex values into a regular grid
    vertex_values = T_fn.compute_vertex_values(_mesh)
    grid_size = MESH_N + 1
    field = vertex_values[_sort_idx].reshape((grid_size, grid_size)).astype(np.float32)

    return T_fn, field


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_plot(field: np.ndarray, param_row: np.ndarray, path: str) -> None:
    k_val, Q_val, T_left, T_right = param_row
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                   cmap="hot", aspect="equal")
    plt.colorbar(im, ax=ax, label="Temperature (K)")
    ax.set_title(
        f"k={k_val:.1f} W/m·K,  Q={Q_val:.0f} W/m³\n"
        f"T_left={T_left:.0f} K,  T_right={T_right:.0f} K"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOT_DIR,      exist_ok=True)
    os.makedirs(VTK_DIR,       exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- Latin Hypercube Sampling ---
    sampler   = LatinHypercube(d=4, seed=RANDOM_SEED)
    lhs_unit  = sampler.random(n=N_SAMPLES)                        # (N, 4) in [0, 1]
    params    = scale(lhs_unit, PARAM_BOUNDS_LOW, PARAM_BOUNDS_HIGH).astype(np.float32)

    grid_size = MESH_N + 1
    fields    = np.zeros((N_SAMPLES, grid_size, grid_size), dtype=np.float32)

    print(f"Generating {N_SAMPLES} solutions on {MESH_N}×{MESH_N} mesh "
          f"({grid_size}×{grid_size} vertex grid)...")

    for i in range(N_SAMPLES):
        k_val, Q_val, T_left, T_right = params[i]

        T_fn, field = solve_heat_conduction(k_val, Q_val, T_left, T_right)
        fields[i] = field

        # Save preview outputs for first 5 samples only
        if i < 5:
            save_plot(field, params[i], os.path.join(PLOT_DIR, f"sample_{i:04d}.png"))
            vtk_file = File(os.path.join(VTK_DIR, f"sample_{i:04d}.pvd"))
            vtk_file << T_fn

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:>4}/{N_SAMPLES}]  k={k_val:5.1f}  Q={Q_val:6.0f}  "
                  f"T_left={T_left:.0f}  T_right={T_right:.0f}  "
                  f"T=[{field.min():.1f}, {field.max():.1f}] K")

    # --- Save full dataset ---
    np.savez_compressed(
        DATASET_PATH,
        params            = params,
        fields            = fields,
        lhs_samples       = lhs_unit.astype(np.float32),
        param_names       = np.array(PARAM_NAMES),
        param_bounds_low  = PARAM_BOUNDS_LOW,
        param_bounds_high = PARAM_BOUNDS_HIGH,
    )
    print(f"\nDataset saved: {DATASET_PATH}")
    print(f"  params shape : {params.shape}")
    print(f"  fields shape : {fields.shape}")

    # --- Train / val / test split (80 / 10 / 10) ---
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(N_SAMPLES)
    n_train = int(0.8 * N_SAMPLES)   # 400
    n_val   = int(0.1 * N_SAMPLES)   #  50

    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train : n_train + n_val],
        "test":  idx[n_train + n_val :],
    }

    for split_name, split_idx in splits.items():
        path = os.path.join(PROCESSED_DIR, f"{split_name}.npz")
        np.savez_compressed(
            path,
            params = params[split_idx],
            fields = fields[split_idx],
        )
        print(f"  {split_name:5s} ({len(split_idx):>3} samples) → {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
