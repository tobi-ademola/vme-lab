"""
validate_solver.py
==================
Solver Credibility Checks — 2D Transient Heat Conduction

Three independent checks:
  1. Analytical convergence  — L2 error vs known solution < 2 % at every timestep
  2. Energy balance          — domain energy must rise monotonically when Q > 0
                               and BCs allow net heat in
  3. Visual sanity           — 2×3 grid of field snapshots saved to PNG

Run directly:
    python3 validate_solver.py
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from fenics import (
    UnitSquareMesh, FunctionSpace, Function, Constant, Expression,
    TrialFunction, TestFunction, DirichletBC, near,
    dot, grad, dx, ds, solve, assemble,
    DOLFIN_EPS,
)

# Import the solver components we are validating
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_transient import (
    _mesh, _V, _sort_idx,
    GRID, MESH_N, DT, N_STEPS,
    field_to_grid,
    solve_transient,
    make_variational_form,
    make_boundary_conditions,
    make_initial_condition,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH  = os.path.join(SCRIPT_DIR, "validation_plots.png")

# ---------------------------------------------------------------------------
# Check 1 — Analytical convergence
# ---------------------------------------------------------------------------
# Known solution for the 1D heat equation on [0,1]:
#
#   ∂T/∂t = α ∂²T/∂x²    with  α = k / ρcₚ
#
#   BCs : T(0,t) = T(1,t) = 0       (homogeneous Dirichlet)
#   IC  : T(x,0) = sin(πx)          (first mode of the sine series)
#
#   Exact solution:
#       T(x,t) = sin(πx) · exp(−π²αt)
#
# We lift the homogeneous problem to the actual boundary values by setting
# T_left = T_right = 0 (the solver always adds T_left/T_right, so we use 300 K
# and subtract it before computing the error).
#
# The L2 error between the FEniCS solution and the analytical solution is
# computed at every timestep and must stay below 2 %.

def analytical_solution(x: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """
    Exact 1D solution T(x,t) = sin(πx) · exp(−π²αt).

    This satisfies:  ∂T/∂t = α ∂²T/∂x²  with T(0)=T(1)=0, T(x,0)=sin(πx).

    Parameters
    ----------
    x     : 1D coordinate array in [0, 1]
    t     : time [s]
    alpha : thermal diffusivity k / ρcₚ  [m²/s]

    Returns
    -------
    T : np.ndarray, same shape as x
    """
    return np.sin(np.pi * x) * np.exp(-(np.pi ** 2) * alpha * t)


def check_analytical_convergence() -> bool:
    """
    Run a single known-solution case and assert L2 error < 2 % at every timestep.

    Strategy
    --------
    The FEniCS solver lives in 2D (x, y) but the analytical solution is 1D
    (depends only on x).  We compare along the centreline y = 0.5 by
    extracting the middle row of each snapshot.

    Test parameters
    ---------------
    k      = 1.0 W/m·K
    ρcₚ    = 1.0 J/m³·K   → α = k/ρcₚ = 1.0 m²/s  (fast diffusion, easy to verify)
    Q      = 0.0 W/m³      (no source — keeps the PDE linear with known solution)
    T_left = T_right = 0 K (homogeneous BCs — matched to the analytical solution)

    The IC is set to sin(πx) via a custom Expression rather than through the
    IC dispatcher (which doesn't include a sine IC) because this check needs
    the exact analytical IC.

    Returns
    -------
    True if all timestep errors are below the 2 % threshold, False otherwise.
    """
    print("\n--- Check 1: Analytical convergence ---")

    k_val  = 1.0
    rho_cp = 1.0
    alpha  = k_val / rho_cp
    Q_val  = 0.0
    T_bc   = 0.0   # homogeneous Dirichlet

    # Sine IC: T(x,y,0) = sin(πx)
    T_n = Function(_V)
    T_n.interpolate(Expression("sin(pi * x[0])", degree=4, pi=np.pi))

    bcs = [
        DirichletBC(_V, Constant(T_bc), lambda x, ob: ob and near(x[0], 0.0)),
        DirichletBC(_V, Constant(T_bc), lambda x, ob: ob and near(x[0], 1.0)),
    ]

    # x-coordinates along the middle row of the sorted grid
    xs = np.linspace(0.0, 1.0, GRID)

    errors = []
    passed = True

    for step in range(1, N_STEPS + 1):
        t = step * DT

        T_trial = TrialFunction(_V)
        v       = TestFunction(_V)

        # Use "dirichlet_lr" config so no extra boundary terms are added
        a, L = make_variational_form(
            T_trial, v, T_n, k_val, Q_val, rho_cp, DT, "dirichlet_lr"
        )

        T_new = Function(_V)
        solve(a == L, T_new, bcs)
        T_n.assign(T_new)

        grid = field_to_grid(T_n)          # (33, 33)
        row  = grid[GRID // 2, :]          # centreline y ≈ 0.5, shape (33,)

        exact  = analytical_solution(xs, t, alpha)
        l2_num = np.sqrt(np.mean((row - exact) ** 2))
        l2_den = np.sqrt(np.mean(exact ** 2)) + 1e-12
        rel_err = l2_num / l2_den

        errors.append(rel_err)
        status = "OK" if rel_err < 0.02 else "FAIL"
        if rel_err >= 0.02:
            passed = False
        print(f"  t={t:.2f} s   rel L2 error = {rel_err:.4f}   [{status}]")

    max_err = max(errors)
    print(f"\n  Max relative L2 error across all timesteps: {max_err:.4f}")
    if passed:
        print("  PASS — all timesteps below 2 % threshold.")
    else:
        print("  FAIL — one or more timesteps exceeded 2 % threshold.")
    return passed


# ---------------------------------------------------------------------------
# Check 2 — Energy balance
# ---------------------------------------------------------------------------
# When Q > 0 and boundary conditions allow net heat to enter the domain,
# the total thermal energy E(t) = ∫_Ω ρcₚ T dΩ must increase monotonically.
#
# We use a case with:
#   - Neumann (insulated) BCs on all sides → no boundary heat loss
#   - Q > 0 → constant heat injection
#   - IC: uniform T_0 = 300 K
#
# Under these conditions:  dE/dt = Q · |Ω| = Q · 1.0  (since |Ω| = 1)
# So E must strictly increase at every timestep.

def check_energy_balance() -> bool:
    """
    Verify that total domain energy increases monotonically when Q>0 and BCs
    are fully insulating (pure Neumann on all sides).

    Pure Neumann BCs are not directly available in BC_CONFIGS (which all have
    at least one Dirichlet wall) so we build the special case manually here.

    Returns
    -------
    True if energy is strictly monotonically increasing, False otherwise.
    """
    print("\n--- Check 2: Energy balance ---")

    k_val  = 10.0
    rho_cp = 1_000_000.0
    Q_val  = 500.0
    T_init = 300.0

    # IC: uniform temperature
    T_n = Function(_V)
    T_n.assign(Constant(T_init))

    # Pure Neumann (no Dirichlet) — zero-flux on all boundaries.
    # This makes the system singular unless we pin one DOF; for energy-balance
    # testing we accept the relative change rather than absolute values.
    bcs = []

    energies = [float(assemble(T_n * dx)) * rho_cp]

    monotone = True
    for step in range(1, N_STEPS + 1):
        T_trial = TrialFunction(_V)
        v       = TestFunction(_V)
        a_form  = (rho_cp / DT) * T_trial * v * dx + Constant(k_val) * dot(grad(T_trial), grad(v)) * dx
        L_form  = (rho_cp / DT) * T_n * v * dx + Constant(Q_val) * v * dx

        T_new = Function(_V)
        try:
            solve(a_form == L_form, T_new, bcs)
        except Exception:
            # Pure Neumann can be singular; pin using LU with null-space handling
            # For the purpose of this energy check we rely on the integral trend.
            pass

        T_n.assign(T_new)
        E = float(assemble(T_n * dx)) * rho_cp
        energies.append(E)

        delta = energies[-1] - energies[-2]
        status = "OK" if delta > 0 else "FAIL"
        if delta <= 0:
            monotone = False
        print(f"  step {step:>2}  E = {E:.2f} J   ΔE = {delta:+.4f}   [{status}]")

    if monotone:
        print("  PASS — energy increased at every timestep.")
    else:
        print("  FAIL — energy was not strictly monotonically increasing.")
    return monotone


# ---------------------------------------------------------------------------
# Check 3 — Visual snapshot grid
# ---------------------------------------------------------------------------
# Produce a 2×3 figure:  rows = two contrasting samples,
# columns = t=0, t=T_end/2, t=T_end.
# Saved to validation_plots.png.

SAMPLE_A = dict(
    k_val=1.0, Q_val=1000.0, T_left=600.0, T_right=300.0,
    rho_cp=500_000.0, ic_type="gaussian_hotspot", bc_config="dirichlet_lr",
    label="High Q, low k, Gaussian IC",
)
SAMPLE_B = dict(
    k_val=50.0, Q_val=0.0, T_left=300.0, T_right=600.0,
    rho_cp=4_000_000.0, ic_type="uniform", bc_config="dirichlet_all",
    label="High k, zero Q, uniform IC",
)


def check_visual_snapshots() -> bool:
    """
    Run two contrasting samples and save a 2×3 snapshot grid to PNG.

    Columns: t=0, t=T_end/2, t=T_end.
    Rows   : Sample A (high Q, low k), Sample B (high k, zero Q).

    Returns
    -------
    True always — this is a visual/manual check, not a pass/fail assertion.
    """
    print("\n--- Check 3: Visual snapshot grid ---")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    col_steps = [0, N_STEPS // 2, N_STEPS]
    col_times = [s * DT for s in col_steps]

    for row_idx, sample in enumerate([SAMPLE_A, SAMPLE_B]):
        label     = sample.pop("label")
        snapshots = solve_transient(**sample)
        sample["label"] = label   # restore so dict is unmodified

        if snapshots is None:
            print(f"  Row {row_idx}: solve_transient returned None — "
                  f"complete the TODO blocks in generate_transient.py first.")
            plt.close()
            return False

        vmin, vmax = snapshots.min(), snapshots.max()

        for col_idx, (step, t) in enumerate(zip(col_steps, col_times)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                snapshots[step], origin="lower", extent=[0, 1, 0, 1],
                cmap="hot", aspect="equal", vmin=vmin, vmax=vmax,
            )
            ax.set_title(f"t = {t:.1f} s", fontsize=9)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)
            plt.colorbar(im, ax=ax, label="T (K)", shrink=0.8)

        axes[row_idx, 0].set_ylabel(f"{label}\ny", fontsize=8)

    plt.suptitle("Solver Validation — Field Snapshots", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOT_PATH}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = {}

    results["analytical_convergence"] = check_analytical_convergence()
    results["energy_balance"]         = check_energy_balance()
    results["visual_snapshots"]       = check_visual_snapshots()

    print("\n=== Validation Summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}  {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nOne or more checks failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
