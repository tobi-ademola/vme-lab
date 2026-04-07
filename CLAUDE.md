# CLAUDE.md — AI-Powered Virtual Mechanical Engineering Laboratory

## Project overview

This is a final-year dissertation project by Ademola Oluwatobi, Department of Mechanical
Engineering, Edo State University, Iyamho, Nigeria. Supervisor: Mr Jeremiah Akhimien.

The platform is a web-based virtual laboratory where neural surrogate models serve as
real-time physics engines. Students interact with five mechanical engineering experiments
through a standard browser — no installation, no specialist hardware, no licensing fees.

The project directly addresses two problems:

- Inadequate physical laboratory infrastructure in Nigerian universities
- Existing virtual lab platforms are pre-computed and cannot support real-time parametric
  interaction

---

## Repository structure

```
vme-lab/
├── data/
│   ├── raw/
│   │   ├── heat_conduction/      ← FEniCS output (.npz datasets)
│   │   ├── fin_array/            ← Analytical solver output
│   │   ├── beam_bending/         ← FEniCS output
│   │   ├── pipe_flow/            ← OpenFOAM output
│   │   └── bluff_body/           ← OpenFOAM output
│   └── processed/                ← Normalised, split datasets ready for training
├── models/
│   ├── heat_conduction/          ← Trained FNO (.pt and .onnx)
│   ├── fin_array/                ← Trained DNN (.pt and .onnx)
│   ├── beam_bending/             ← Trained DNN (.pt and .onnx)
│   ├── pipe_flow/                ← Trained FNO (.pt and .onnx)
│   └── bluff_body/               ← Trained FNO (.pt and .onnx)
├── notebooks/                    ← Google Colab training notebooks
├── backend/                      ← FastAPI application
│   ├── main.py
│   ├── routers/
│   ├── schemas/
│   ├── inference/                ← ONNX Runtime inference wrappers
│   └── requirements.txt
├── frontend/                     ← React application
│   ├── src/
│   │   ├── components/
│   │   ├── experiments/          ← One folder per experiment module
│   │   └── visualisation/        ← D3.js field rendering components
│   └── package.json
└── CLAUDE.md
```

---

## The five experiment modules

| #   | Experiment                      | Solver            | ML architecture | Primary output            |
| --- | ------------------------------- | ----------------- | --------------- | ------------------------- |
| 1   | 2D steady-state heat conduction | FEniCS            | FNO             | T(x,y) field              |
| 2   | Fin array heat transfer         | Python analytical | DNN             | T(x), η_f, Q̇              |
| 3   | Beam bending and stress         | FEniCS            | DNN             | w(x), M(x), σ_max         |
| 4   | Internal pipe flow              | OpenFOAM          | FNO             | u(r), f, ΔP, Re           |
| 5   | 2D flow over bluff body         | OpenFOAM          | FNO             | u field, C_D, C_L, C_p(θ) |

**Implementation priority order:** 1 → 5 → 2 → 3 → 4

Heat conduction is first because it is simplest and validates the full pipeline end-to-end.
Bluff body is second because it directly addresses the wind tunnel gap at the institution.

---

## System architecture

```
[React Frontend]  ←→  [FastAPI Backend]  ←→  [ONNX Runtime]  ←→  [.onnx model files]
     D3.js                  Pydantic                CPU inference
     Axios               CORS middleware
```

Training pipeline (offline, Google Colab):

```
Latin Hypercube Sampling → Solver (FEniCS/OpenFOAM) → Dataset (.npz) → PyTorch training → ONNX export
```

---

## Technology stack

### Data generation

- **FEniCS 2019.1.0** (legacy) — heat conduction, beam bending
- **OpenFOAM** — pipe flow, bluff body
- **Python analytical** — fin array
- **pyDOE2 / scipy.stats.qmc** — Latin Hypercube Sampling

### Machine learning

- **PyTorch** — model definition and training
- **NVIDIA PhysicsNeMo** — FNO implementation
- **Google Colab Pro** — GPU training (no local GPU available)
- **ONNX** — model export format

### Backend

- **FastAPI** — REST API framework
- **ONNX Runtime** — CPU inference serving
- **Pydantic** — request/response validation

### Frontend

- **React** — UI framework
- **D3.js** — 2D field visualisation (heatmaps, contour plots, vector fields)
- **Axios** — HTTP client

### Infrastructure

- **Docker** — containerisation
- **Git / GitHub** — version control

---

## Development environment

- **OS:** Windows 11 Pro with WSL2 (Ubuntu 24.04.4 LTS)
- **Machine:** Dell Latitude 5510, Intel i5-10310U, 8GB RAM, Intel UHD Graphics (no GPU)
- **Python:** System Python 3 inside WSL2
- **Editor:** VS Code connected to WSL2 via Remote-WSL extension
- **GPU training:** Google Colab Pro (remote)

### Key environment facts Claude must respect

- There is **no display/GUI** in WSL2 — never write code that opens a matplotlib window,
  calls `plot()` without saving to file, or calls `interactive()`
- Always use `matplotlib.use('Agg')` before importing `matplotlib.pyplot`
- FEniCS plots must be saved as PNG via `plt.savefig()`, not displayed
- VTK output (`File('output.pvd')`) is fine — ParaView is installed on Windows and reads
  WSL files via `\\wsl$\Ubuntu\...`
- GPU-dependent code (PyTorch training with CUDA) belongs in Colab notebooks, not local scripts

---

## Common commands

### FEniCS data generation

```bash
cd ~/vme-lab
python3 data/raw/heat_conduction/generate.py
```

### Run FastAPI backend (development)

```bash
cd ~/vme-lab/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run React frontend (development)

```bash
cd ~/vme-lab/frontend
npm run dev
```

### Run both together

```bash
# Terminal 1
cd ~/vme-lab/backend && uvicorn main:app --reload

# Terminal 2
cd ~/vme-lab/frontend && npm run dev
```

### Install Python dependencies

```bash
pip3 install <package> --break-system-packages
```

### Check FEniCS is working

```bash
python3 -c "from fenics import *; print('FEniCS OK')"
```

---

## Coding conventions

### Python

- Use **f-strings** for all string formatting
- Use **type hints** on all function signatures
- Use **numpy** for all array operations — never plain Python lists for numerical data
- Dataset files are saved as `.npz` (numpy compressed) with consistent key names:
  - `params` — input parameter array, shape `(N, num_params)`
  - `fields` — output field array, shape `(N, nx, ny)` for 2D fields
  - `scalars` — scalar output array, shape `(N, num_scalars)` where applicable
- All random seeds set to `42` for reproducibility

### FEniCS scripts

- Always create `output/plots/` and `output/vtk/` directories with `os.makedirs(..., exist_ok=True)`
- Always save VTK alongside PNG — VTK for verification in ParaView, PNG for quick checks
- Print error metrics at every timestep
- Never import from sympy in FEniCS scripts — sympy symbols silently break FEniCS expressions

### FastAPI

- All routes go in `backend/routers/` — one file per experiment
- All request/response models go in `backend/schemas/`
- ONNX Runtime sessions are loaded once at startup, not per-request
- Always include CORS middleware configured for the React dev server

### React / Frontend

- One folder per experiment inside `src/experiments/`
- D3.js field renderers go in `src/visualisation/` and are experiment-agnostic
- API calls go through a centralised `src/api/client.js` using Axios
- Parameter input components are shared across experiments where possible

---

## Data generation conventions

### Latin Hypercube Sampling

- Always use `pyDOE2.lhs()` or `scipy.stats.qmc.LatinHypercube`
- Minimum **500 samples** per experiment for FNO training
- Minimum **200 samples** per experiment for DNN training
- 80/10/10 train/validation/test split
- Save the LHS sample array alongside the dataset for reproducibility

### Parameter ranges (heat conduction — Experiment 1)

| Parameter            | Symbol  | Range         | Units |
| -------------------- | ------- | ------------- | ----- |
| Thermal conductivity | k       | [1.0, 50.0]   | W/m·K |
| Heat source          | Q       | [0.0, 1000.0] | W/m³  |
| Left boundary temp   | T_left  | [300, 600]    | K     |
| Right boundary temp  | T_right | [300, 600]    | K     |

_(Ranges for other experiments to be added as implemented)_

---

## Current implementation status

- [x] WSL2 + Ubuntu 24.04 set up
- [x] FEniCS 2019.1.0 installed and verified
- [x] Project directory structure created
- [x] Poisson equation test case working
- [x] Time-dependent heat equation working (WSL-compatible, VTK output)
- [x] Parametric heat conduction data generation script
- [x] Latin Hypercube Sampling pipeline
- [ ] FNO training notebook (Colab)
- [ ] ONNX export pipeline
- [ ] FastAPI backend skeleton
- [ ] ONNX Runtime inference wrapper
- [ ] React frontend skeleton
- [ ] D3.js heat field renderer
- [ ] Full end-to-end prototype (Experiment 1)
- [ ] Remaining four experiments

---

## Known issues and gotchas

1. **FEniCS JIT compilation** prints "Calling FFC just-in-time compiler" multiple times
   on first run — this is normal, not an error. Subsequent runs are faster
2. **`fenics.__version__`** raises AttributeError — use `dolfin.__version__` instead
3. **matplotlib must use Agg backend** before any pyplot import in WSL — place
   `matplotlib.use('Agg')` immediately after `import matplotlib`
4. **pip install in Ubuntu 24.04** requires `--break-system-packages` flag due to PEP 668

---

## Dissertation context

- **Institution:** Edo State University, Iyamho, Nigeria
- **Department:** Mechanical Engineering
- **Supervisor:** Mr Jeremiah Akhimien
- **Evaluation method:** Accuracy benchmarking (surrogate vs solver) + System Usability
  Scale (SUS) usability study
- **Target users:** Mechanical engineering students with no CFD/FEM experience
- **Deadline:** Working prototype in 3–4 months from project start
