"""
generate_transient.py
=====================
2D Transient Heat Conduction — Parametric Dataset Generator

Strong-form PDE
---------------
    ρcₚ ∂T/∂t = ∇·(k ∇T) + Q    on Ω = [0,1]²,  t ∈ (0, T_end]
    T(x, 0)   = T₀(x)            initial condition (varies per sample)
    + boundary conditions         (varies per sample — see BC_CONFIGS below)

This script generates N_SAMPLES solutions, each consisting of N_STEPS+1
time snapshots (including t=0), saved as a single compressed .npz file:

    fields  : (N_SAMPLES, N_STEPS+1, GRID, GRID)  float32   — T(x,y,t) snapshots
    params  : (N_SAMPLES, N_PARAMS)                float32   — physical parameters
    t_grid  : (N_STEPS+1,)                         float32   — time coordinate vector

Learning-Oriented Scaffold
--------------------------
Boilerplate (imports, LHS, file I/O, outer loop, grid interpolation) is
written for you.  The physically and mathematically significant sections are
left as # TODO blocks.  Each TODO block states:
  (a) the governing mathematics
  (b) the relevant FEniCS API classes/functions to use
  (c) the expected shape or value to check your work against

Read the docstring above each TODO before filling it in.
"""

# ---------------------------------------------------------------------------
# Imports — do not modify
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale as lhs_scale
from fenics import (
    UnitSquareMesh, FunctionSpace, Function, Constant, Expression,
    TrialFunction, TestFunction,
    DirichletBC, near,
    dot, grad, dx, solve,
    File, assign,
    DOLFIN_EPS,
)

# ---------------------------------------------------------------------------
# Configuration — adjust these before running
# ---------------------------------------------------------------------------

N_SAMPLES   = 300        # total number of parametric samples to generate
MESH_N      = 32         # 32×32 elements → 33×33 vertex grid
N_STEPS     = 20         # number of Backward-Euler time steps
DT          = 0.1        # time step size (seconds)
RANDOM_SEED = 42

GRID        = MESH_N + 1          # 33

# Parameter names and bounds:
#   k       — thermal conductivity   [W/m·K]
#   Q       — volumetric heat source [W/m³]
#   T_left  — left-wall temperature  [K]
#   T_right — right-wall temperature [K]
#   rho_cp  — volumetric heat capacity ρcₚ [J/m³·K]  (density × specific heat)
PARAM_NAMES       = ["k", "Q", "T_left", "T_right", "rho_cp"]
PARAM_BOUNDS_LOW  = np.array([  1.0,    0.0, 300.0, 300.0,  500_000.0])
PARAM_BOUNDS_HIGH = np.array([ 50.0, 1000.0, 600.0, 600.0, 4_000_000.0])
N_PARAMS          = len(PARAM_NAMES)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR     = os.path.join(SCRIPT_DIR, "output", "transient_plots")
VTK_DIR      = os.path.join(SCRIPT_DIR, "output", "transient_vtk")
DATASET_PATH = os.path.join(SCRIPT_DIR, "heat_conduction_transient.npz")
PROCESSED_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "processed", "heat_conduction_transient"
)

# ---------------------------------------------------------------------------
# Mesh — built once, reused for all solves
# ---------------------------------------------------------------------------

_mesh     = UnitSquareMesh(MESH_N, MESH_N)
_V        = FunctionSpace(_mesh, "P", 1)
_coords   = _mesh.coordinates()                           # (n_verts, 2)
_sort_idx = np.lexsort((_coords[:, 0], _coords[:, 1]))   # row-major: x fastest


# ---------------------------------------------------------------------------
# Grid extraction helper — do not modify
# ---------------------------------------------------------------------------

def field_to_grid(T_fn: Function) -> np.ndarray:
    """Extract a FEniCS Function to a (GRID, GRID) numpy array.

    Vertex values from FEniCS are not guaranteed to be in any particular
    spatial order.  _sort_idx reorders them by (y, x) so that the result
    can be reshaped into a regular grid without scrambling spatial structure.

    Parameters
    ----------
    T_fn : FEniCS Function (P1 on _mesh)

    Returns
    -------
    np.ndarray of shape (GRID, GRID) = (33, 33), dtype float32
    """
    vals = T_fn.compute_vertex_values(_mesh)
    return vals[_sort_idx].reshape((GRID, GRID)).astype(np.float32)


# ---------------------------------------------------------------------------
# Initial condition selector — boilerplate dispatcher written for you
# ---------------------------------------------------------------------------

IC_TYPES = ["uniform", "gaussian_hotspot", "linear_ramp"]


def make_initial_condition(ic_type: str, T_left: float, T_right: float) -> Function:
    """Create and return a FEniCS Function representing the initial temperature field.

    Parameters
    ----------
    ic_type : one of IC_TYPES
    T_left  : left-wall temperature (K) — used to anchor the field
    T_right : right-wall temperature (K) — used to anchor the field

    Returns
    -------
    T0 : Function on _V with values set to the chosen IC
    """
    T0 = Function(_V)

    if ic_type == "uniform":
        _set_ic_uniform(T0, T_right)

    elif ic_type == "gaussian_hotspot":
        _set_ic_gaussian_hotspot(T0, T_left, T_right)

    elif ic_type == "linear_ramp":
        _set_ic_linear_ramp(T0, T_left, T_right)

    else:
        raise ValueError(f"Unknown IC type: {ic_type!r}.  Choose from {IC_TYPES}.")

    return T0


# ── IC implementations ────────────────────────────────────────────────────


def _set_ic_uniform(T0: Function, T_ref: float) -> None:
    """
    Uniform initial temperature equal to the right-wall value.

    Mathematical expression
    -----------------------
        T(x, y, 0) = T_ref     ∀ (x, y) ∈ Ω

    This is the simplest IC — the domain starts at a single temperature and
    evolves toward the steady-state distribution driven by BCs and Q.

    # TODO ─────────────────────────────────────────────────────────────────
    Assign a constant initial temperature T_ref to the FEniCS Function T0.

    FEniCS API hints
    ----------------
    - T0.interpolate(Expression(..., degree=0))
      OR
    - T0.assign(Constant(T_ref))
      (Constant is already imported at the top of this file)

    Expected result
    ---------------
    After this function returns, T0.vector()[:] should all equal T_ref.
    Quick check:
        assert abs(T0.vector()[:].min() - T_ref) < 1e-6
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


def _set_ic_gaussian_hotspot(T0: Function, T_left: float, T_right: float) -> None:
    """
    Gaussian hot spot superimposed on a background temperature.

    Mathematical expression
    -----------------------
        T_bg  = (T_left + T_right) / 2                  background temperature
        A     = 0.3 * T_bg                               amplitude of hot spot
        σ     = 0.1                                      width parameter
        T(x, y, 0) = T_bg + A · exp(−[(x−0.5)² + (y−0.5)²] / (2σ²))

    This IC creates a concentrated hot region at the domain centre.  It tests
    the surrogate's ability to capture localised features evolving over time.

    # TODO ─────────────────────────────────────────────────────────────────
    Interpolate the Gaussian expression onto T0.

    FEniCS API hints
    ----------------
    - Use Expression with degree=2 (quadratic interpolation keeps the Gaussian
      shape — degree=1 would over-smooth it).
    - Access coordinates with x[0] (x-coordinate) and x[1] (y-coordinate)
      inside the Expression string.
    - Pass Python values as keyword arguments to Expression:
        Expression("...", degree=2, T_bg=T_bg, A=A, sigma=0.1)

    Expected result
    ---------------
    T0.vector()[:].max() should be noticeably above T_bg (roughly T_bg + A).
    T0.vector()[:].min() should be close to T_bg.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


def _set_ic_linear_ramp(T0: Function, T_left: float, T_right: float) -> None:
    """
    Linear temperature ramp from T_left (x=0) to T_right (x=1).

    Mathematical expression
    -----------------------
        T(x, y, 0) = T_left + (T_right − T_left) · x

    This IC is the analytical steady-state solution for a 1D rod with no heat
    source (Q=0) and Dirichlet BCs only.  Starting from this IC with Q>0
    reveals how a source term perturbs the steady state — pedagogically useful.

    # TODO ─────────────────────────────────────────────────────────────────
    Interpolate the linear ramp onto T0.

    FEniCS API hints
    ----------------
    - Use Expression with degree=1 (the ramp is linear, so degree=1 is exact).
    - The x-coordinate inside Expression strings is x[0].
    - Pass T_left and T_right as keyword arguments.

    Expected result
    ---------------
    T0.vector()[:].min() ≈ min(T_left, T_right)
    T0.vector()[:].max() ≈ max(T_left, T_right)
    The field should look like a left-to-right colour gradient.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# Boundary condition configurations — boilerplate dispatcher written for you
# ---------------------------------------------------------------------------

BC_CONFIGS = [
    "dirichlet_lr",          # Dirichlet left & right,  Neumann top & bottom
    "dirichlet_all",         # Dirichlet on all four walls
    "neumann_top_flux",      # Dirichlet left & right,  fixed flux on top, insulated bottom
    "robin_right",           # Dirichlet left,  Robin (convective) on right,  insulated top & bottom
    "insulated_sides_source",# Neumann (zero) on left & right,  Dirichlet top & bottom
]


def make_boundary_conditions(
    bc_config: str,
    T_left: float,
    T_right: float,
) -> list:
    """Return a list of FEniCS DirichletBC objects for the chosen configuration.

    Neumann (flux) and Robin conditions are applied weakly — they modify the
    variational form (a, L) rather than appearing in this list.  See
    make_variational_form() for where those terms are added.

    Parameters
    ----------
    bc_config : one of BC_CONFIGS
    T_left    : left-wall temperature (K)
    T_right   : right-wall temperature (K)

    Returns
    -------
    list of DirichletBC objects (may be empty if all BCs are weak)
    """
    if bc_config == "dirichlet_lr":
        return _bc_dirichlet_lr(T_left, T_right)
    elif bc_config == "dirichlet_all":
        return _bc_dirichlet_all(T_left, T_right)
    elif bc_config == "neumann_top_flux":
        return _bc_neumann_top_flux(T_left, T_right)
    elif bc_config == "robin_right":
        return _bc_robin_right(T_left, T_right)
    elif bc_config == "insulated_sides_source":
        return _bc_insulated_sides_source(T_left, T_right)
    else:
        raise ValueError(f"Unknown BC config: {bc_config!r}.  Choose from {BC_CONFIGS}.")


# ── BC implementations ────────────────────────────────────────────────────


def _bc_dirichlet_lr(T_left: float, T_right: float) -> list:
    """
    Dirichlet on left and right walls; Neumann (insulated) on top and bottom.

    Mathematical statement
    ----------------------
        T = T_left   on x = 0
        T = T_right  on x = 1
        ∂T/∂n = 0    on y = 0  and  y = 1   (natural / do-nothing BC)

    The zero-flux Neumann condition on top and bottom is automatic in FEniCS —
    you do NOT need to add boundary integral terms to the variational form for
    these edges.  Only non-zero fluxes and Robin conditions need weak treatment.

    # TODO ─────────────────────────────────────────────────────────────────
    Create and return two DirichletBC objects.

    FEniCS API hints
    ----------------
    DirichletBC(_V, Constant(value), marker_function)
    where marker_function is a lambda:
        lambda x, on_boundary: on_boundary and near(x[0], 0.0)
    for the left wall (x = 0).

    Both `near` and `Constant` are imported at the top of this file.

    Expected result
    ---------------
    Returns a list of length 2.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


def _bc_dirichlet_all(T_left: float, T_right: float) -> list:
    """
    Dirichlet temperature prescribed on all four walls.

    Mathematical statement
    ----------------------
        T = T_left   on x = 0
        T = T_right  on x = 1
        T = T_right  on y = 0   (bottom wall held at T_right)
        T = T_left   on y = 1   (top wall held at T_left)

    Having four Dirichlet walls creates a more constrained problem —
    the domain temperature is "pinned" at all edges, so only Q drives
    the interior above or below the boundary values.

    # TODO ─────────────────────────────────────────────────────────────────
    Create and return four DirichletBC objects (one per wall).

    FEniCS API hints
    ----------------
    - Left  wall: near(x[0], 0.0)
    - Right wall: near(x[0], 1.0)
    - Bottom wall: near(x[1], 0.0)
    - Top wall:    near(x[1], 1.0)

    Expected result
    ---------------
    Returns a list of length 4.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


def _bc_neumann_top_flux(T_left: float, T_right: float) -> list:
    """
    Dirichlet left and right; fixed outward heat flux on top; insulated bottom.

    Mathematical statement
    ----------------------
        T = T_left              on x = 0
        T = T_right             on x = 1
        −k ∂T/∂n = q_top = 200 W/m²   on y = 1  (heat leaving through top)
        ∂T/∂n = 0               on y = 0  (insulated bottom, natural BC)

    The non-zero Neumann flux on y=1 must be included in the linear form L
    as a boundary integral:  L += -q_top * v * ds(top)
    See the make_variational_form() TODO for details.

    This function only returns the Dirichlet BCs — the Neumann term is handled
    in make_variational_form().

    # TODO ─────────────────────────────────────────────────────────────────
    Create and return the two Dirichlet BCs (left and right walls only).

    FEniCS API hints
    ----------------
    Same as _bc_dirichlet_lr — only the left and right walls get Dirichlet BCs.

    Expected result
    ---------------
    Returns a list of length 2.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


def _bc_robin_right(T_left: float, T_right: float) -> list:
    """
    Dirichlet left wall; Robin (convective) condition on right; insulated top & bottom.

    Mathematical statement
    ----------------------
        T = T_left                       on x = 0
        −k ∂T/∂n = h(T − T_∞)           on x = 1   (Robin / Newton's law of cooling)
        ∂T/∂n = 0                        on y = 0 and y = 1 (natural)

    Robin parameters used here:
        h   = 10 W/m²·K   (convective heat transfer coefficient)
        T_∞ = 300 K        (ambient temperature)

    The Robin condition contributes both to the bilinear form a (stiffness term h·T·v)
    and the linear form L (load term h·T_∞·v).  See make_variational_form().

    This function only returns the single Dirichlet BC on the left wall.

    # TODO ─────────────────────────────────────────────────────────────────
    Create and return the single DirichletBC for the left wall.

    FEniCS API hints
    ----------------
    DirichletBC(_V, Constant(T_left), lambda x, ob: ob and near(x[0], 0.0))

    Expected result
    ---------------
    Returns a list of length 1.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


def _bc_insulated_sides_source(T_left: float, T_right: float) -> list:
    """
    Zero-flux (insulated) left and right walls; Dirichlet top and bottom.

    Mathematical statement
    ----------------------
        ∂T/∂n = 0    on x = 0 and x = 1  (natural Neumann — no action needed)
        T = T_left   on y = 1  (top)
        T = T_right  on y = 0  (bottom)

    With insulated sides and a volume source Q, heat can only escape through
    top and bottom — it must all flow in the y-direction at steady state.
    Combined with a Gaussian IC, this produces interesting transient patterns.

    # TODO ─────────────────────────────────────────────────────────────────
    Create and return the two Dirichlet BCs on top and bottom walls.

    FEniCS API hints
    ----------------
    - Top wall:    near(x[1], 1.0), value = T_left
    - Bottom wall: near(x[1], 0.0), value = T_right

    Expected result
    ---------------
    Returns a list of length 2.
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# Variational form — one TODO block for the entire weak form
# ---------------------------------------------------------------------------

# Robin parameters (used in _bc_robin_right and here)
H_ROBIN = 10.0    # convective coefficient [W/m²·K]
T_INF   = 300.0   # ambient temperature [K]
Q_NEUMANN_TOP = 200.0  # outward flux through top wall [W/m²]


def make_variational_form(
    T: TrialFunction,
    v: TestFunction,
    T_n: Function,
    k_val: float,
    Q_val: float,
    rho_cp: float,
    dt: float,
    bc_config: str,
) -> tuple:
    """
    Build and return (a, L) — the Backward-Euler variational form.

    Strong-form PDE
    ---------------
        ρcₚ ∂T/∂t = ∇·(k ∇T) + Q

    Backward Euler discretisation in time
    --------------------------------------
    Replace ∂T/∂t ≈ (Tⁿ⁺¹ − Tⁿ) / Δt  (first-order implicit):

        ρcₚ (Tⁿ⁺¹ − Tⁿ) / Δt = ∇·(k ∇Tⁿ⁺¹) + Q

    Rearranged:
        ρcₚ Tⁿ⁺¹ / Δt − ∇·(k ∇Tⁿ⁺¹) = ρcₚ Tⁿ / Δt + Q

    Weak form — multiply by test function v and integrate over Ω,
    then apply Green's identity (integration by parts) to the diffusion term:

        ∫_Ω ρcₚ Tⁿ⁺¹ v / Δt dΩ  +  ∫_Ω k ∇Tⁿ⁺¹ · ∇v dΩ
            + boundary terms (Robin / Neumann)
        =   ∫_Ω ρcₚ Tⁿ v / Δt dΩ  +  ∫_Ω Q v dΩ
            + boundary terms (non-zero Neumann flux / Robin load)

    Parameters
    ----------
    T      : TrialFunction  — Tⁿ⁺¹  (unknown at new time level)
    v      : TestFunction
    T_n    : Function       — Tⁿ  (solution from previous time step)
    k_val  : thermal conductivity  [W/m·K]
    Q_val  : volumetric heat source [W/m³]
    rho_cp : volumetric heat capacity [J/m³·K]
    dt     : time step size [s]
    bc_config : string identifying which BC configuration is active
                (needed to add the correct boundary integral terms)

    Returns
    -------
    a : bilinear form  (left-hand side — stiffness + mass matrix)
    L : linear form    (right-hand side — load vector)

    # TODO ─────────────────────────────────────────────────────────────────
    Implement the Backward-Euler weak form.

    Step-by-step guide
    ------------------
    1. Define FEniCS Constant objects for k_val, Q_val, rho_cp, dt:
           k     = Constant(k_val)
           ...

    2. Write the base bilinear form a (mass term + diffusion term):
           a = (rho_cp/dt) * T * v * dx  +  k * dot(grad(T), grad(v)) * dx

    3. Write the base linear form L (mass term from previous step + source):
           L = (rho_cp/dt) * T_n * v * dx  +  Q_val_c * v * dx

    4. Add boundary-integral terms depending on bc_config:

       "neumann_top_flux": The top wall (y=1) has outward flux q_top = Q_NEUMANN_TOP.
           After integration by parts, this appears in L as:
               L += -Q_NEUMANN_TOP * v * ds
           Use `ds` (boundary measure, already imported as part of FEniCS).
           FEniCS integrates over the ENTIRE boundary with `ds` unless a
           SubDomain/MeshFunction marks specific facets — for simplicity here
           apply it to the full boundary and note this approximation in a comment.

       "robin_right": The right wall (x=1) has condition -k ∂T/∂n = h(T - T_inf).
           After integration by parts, Robin contributes:
               a += H_ROBIN * T * v * ds
               L += H_ROBIN * T_INF * v * ds

    5. Return (a, L).

    FEniCS UFL operators
    --------------------
    - `dx`  — volume integral measure (already imported)
    - `ds`  — boundary integral measure (already imported as part of FEniCS — if
              not imported, add it: from fenics import ds)
    - `dot(grad(T), grad(v))` — inner product of gradients
    - `Constant(value)`       — wraps a scalar for use in UFL expressions

    Expected result
    ---------------
    Both a and L must be UFL Form objects.  You can verify with:
        from ufl import Form
        assert isinstance(a, Form)
        assert isinstance(L, Form)
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# Transient solver — Backward-Euler time-stepping loop
# ---------------------------------------------------------------------------

def solve_transient(
    k_val: float,
    Q_val: float,
    T_left: float,
    T_right: float,
    rho_cp: float,
    ic_type: str,
    bc_config: str,
) -> np.ndarray:
    """
    Run the transient heat equation solver and return all time snapshots.

    Algorithm
    ---------
    1. Create the initial condition T₀ using make_initial_condition()
    2. Create the boundary conditions using make_boundary_conditions()
    3. At each time step n = 0, 1, …, N_STEPS-1:
       a. Build the variational form (a, L) using make_variational_form()
       b. Solve the linear system:  solve(a == L, T_new, bcs)
       c. Advance:  T_n ← T_new
       d. Extract the grid snapshot using field_to_grid()
    4. Return snapshots array of shape (N_STEPS+1, GRID, GRID)

    Parameters
    ----------
    k_val, Q_val, T_left, T_right, rho_cp : physical parameters
    ic_type   : one of IC_TYPES   (selects initial condition)
    bc_config : one of BC_CONFIGS (selects boundary condition set)

    Returns
    -------
    snapshots : np.ndarray, shape (N_STEPS+1, GRID, GRID), dtype float32
                Index 0 is t=0 (initial condition); index N_STEPS is t=T_end.

    # TODO ─────────────────────────────────────────────────────────────────
    Implement the Backward-Euler time-stepping loop.

    Step-by-step guide
    ------------------
    1. Create T_n (solution at previous time step) and T_new (solution at
       next time step) as Function objects on _V.

    2. Initialise T_n from the initial condition:
           T0 = make_initial_condition(ic_type, T_left, T_right)
           T_n.assign(T0)

    3. Allocate the output array:
           snapshots = np.zeros((N_STEPS + 1, GRID, GRID), dtype=np.float32)
           snapshots[0] = field_to_grid(T_n)     # snapshot at t = 0

    4. Get the boundary conditions list:
           bcs = make_boundary_conditions(bc_config, T_left, T_right)

    5. Loop over time steps i = 1 to N_STEPS (inclusive):
       a. T_trial = TrialFunction(_V)
          v       = TestFunction(_V)
          a_form, L_form = make_variational_form(
              T_trial, v, T_n, k_val, Q_val, rho_cp, DT, bc_config
          )
       b. T_new = Function(_V)
          solve(a_form == L_form, T_new, bcs)
       c. T_n.assign(T_new)
          snapshots[i] = field_to_grid(T_n)

    6. Return snapshots.

    FEniCS API hints
    ----------------
    - Function(_V)           — creates a new FEniCS function on the function space
    - TrialFunction(_V)      — unknown in the variational problem
    - TestFunction(_V)       — test function in the variational problem
    - solve(a == L, T, bcs)  — assembles and solves the linear system
    - T_n.assign(T_new)      — copies T_new into T_n (in-place update)

    Expected result
    ---------------
    snapshots.shape == (N_STEPS + 1, GRID, GRID)   →  (21, 33, 33)
    snapshots[0]  — should match the chosen IC (e.g. uniform, Gaussian, ramp)
    snapshots[-1] — should look like the steady-state field for these parameters
    # ───────────────────────────────────────────────────────────────────────
    """
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# Preview plot — boilerplate written for you
# ---------------------------------------------------------------------------

def save_preview(
    snapshots: np.ndarray,
    params_row: np.ndarray,
    ic_type: str,
    bc_config: str,
    path: str,
) -> None:
    """Save a 1×4 strip: t=0, t=T/4, t=T/2, t=T for one sample."""
    step_indices = [0, N_STEPS // 4, N_STEPS // 2, N_STEPS]
    t_vals       = [i * DT for i in step_indices]

    k_val, Q_val, T_left, T_right, rho_cp = params_row
    vmin = snapshots.min()
    vmax = snapshots.max()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, step, t in zip(axes, step_indices, t_vals):
        im = ax.imshow(
            snapshots[step], origin="lower", extent=[0, 1, 0, 1],
            cmap="hot", aspect="equal", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"t = {t:.2f} s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="T (K)")

    fig.suptitle(
        f"k={k_val:.1f} W/m·K  Q={Q_val:.0f} W/m³  "
        f"T_L={T_left:.0f} K  T_R={T_right:.0f} K  ρcₚ={rho_cp:.0f} J/m³·K\n"
        f"IC: {ic_type}   BC: {bc_config}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main — outer loop, LHS sampling, file I/O (boilerplate — do not modify)
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOT_DIR,      exist_ok=True)
    os.makedirs(VTK_DIR,       exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- Latin Hypercube Sampling over 5 parameters ---
    sampler  = LatinHypercube(d=N_PARAMS, seed=RANDOM_SEED)
    lhs_unit = sampler.random(n=N_SAMPLES)                           # (N, 5) in [0,1]
    params   = lhs_scale(lhs_unit, PARAM_BOUNDS_LOW, PARAM_BOUNDS_HIGH).astype(np.float32)

    # Assign IC and BC types by cycling through available options
    ic_assignments  = [IC_TYPES[i % len(IC_TYPES)]   for i in range(N_SAMPLES)]
    bc_assignments  = [BC_CONFIGS[i % len(BC_CONFIGS)] for i in range(N_SAMPLES)]

    # Pre-allocate output arrays
    fields    = np.zeros((N_SAMPLES, N_STEPS + 1, GRID, GRID), dtype=np.float32)
    t_grid    = np.array([i * DT for i in range(N_STEPS + 1)], dtype=np.float32)

    print(
        f"Generating {N_SAMPLES} transient solutions\n"
        f"  Mesh      : {MESH_N}×{MESH_N} elements  ({GRID}×{GRID} vertex grid)\n"
        f"  Time steps: {N_STEPS}  ×  dt={DT} s  →  T_end={N_STEPS*DT:.1f} s\n"
        f"  Output    : fields shape {fields.shape}\n"
    )

    for i in range(N_SAMPLES):
        k_val, Q_val, T_left, T_right, rho_cp = params[i]
        ic_type   = ic_assignments[i]
        bc_config = bc_assignments[i]

        t_start = time.perf_counter()
        snapshots = solve_transient(
            float(k_val), float(Q_val), float(T_left), float(T_right),
            float(rho_cp), ic_type, bc_config,
        )
        elapsed = time.perf_counter() - t_start

        if snapshots is None:
            print(f"  [{i+1:>4}/{N_SAMPLES}] SKIPPED — solve_transient returned None "
                  f"(complete the TODO blocks first)")
            continue

        fields[i] = snapshots

        # Save 1×4 preview strip for first 5 samples
        if i < 5:
            save_preview(
                snapshots, params[i], ic_type, bc_config,
                os.path.join(PLOT_DIR, f"sample_{i:04d}.png"),
            )

        if (i + 1) % 30 == 0 or i == 0:
            T_final_min = snapshots[-1].min()
            T_final_max = snapshots[-1].max()
            print(
                f"  [{i+1:>4}/{N_SAMPLES}]  k={k_val:5.1f}  Q={Q_val:6.0f}  "
                f"ρcₚ={rho_cp:.0e}  IC={ic_type:18s}  BC={bc_config:22s}  "
                f"T_final=[{T_final_min:.0f}, {T_final_max:.0f}] K  "
                f"({elapsed:.1f} s)"
            )

    # --- Save full dataset ---
    np.savez_compressed(
        DATASET_PATH,
        params            = params,
        fields            = fields,
        t_grid            = t_grid,
        lhs_samples       = lhs_unit.astype(np.float32),
        param_names       = np.array(PARAM_NAMES),
        param_bounds_low  = PARAM_BOUNDS_LOW,
        param_bounds_high = PARAM_BOUNDS_HIGH,
        ic_types          = np.array(ic_assignments),
        bc_configs        = np.array(bc_assignments),
    )
    print(f"\nDataset saved → {DATASET_PATH}")
    print(f"  params shape : {params.shape}")
    print(f"  fields shape : {fields.shape}  (N_samples, N_steps+1, {GRID}, {GRID})")

    # --- Train / val / test split (80 / 10 / 10) ---
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(N_SAMPLES)
    n_train = int(0.8 * N_SAMPLES)
    n_val   = int(0.1 * N_SAMPLES)

    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train : n_train + n_val],
        "test":  idx[n_train + n_val :],
    }
    for split_name, split_idx in splits.items():
        path = os.path.join(PROCESSED_DIR, f"{split_name}.npz")
        np.savez_compressed(
            path,
            params    = params[split_idx],
            fields    = fields[split_idx],
            t_grid    = t_grid,
            ic_types  = np.array([ic_assignments[j] for j in split_idx]),
            bc_configs = np.array([bc_assignments[j] for j in split_idx]),
        )
        print(f"  {split_name:5s} ({len(split_idx):>3} samples) → {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
