# AI-Powered Virtual Mechanical Engineering Laboratory

A browser-based virtual lab in which **neural-operator surrogates replace finite-element and CFD
runs**, so field plots update in real time as a student changes parameters. The aim is lab
access for mechanical engineering students at universities with limited physical lab
equipment, with no software installation and no GPU on the student's side.

Project proposal defended March 2026, Edo State University Iyamho, Nigeria.
Guidance: Engr. Jeremiah Akhimien.

## Pipeline

```
Latin Hypercube Sampling ─► FEniCS solver ─► .npz dataset ─► FNO training (PyTorch, Colab GPU) ─► ONNX
                                                                                               │
                         React + D3.js field plots  ◄──  FastAPI  ◄──  ONNX Runtime (CPU)  ◄───┘
```

## Experiment 1: 2D heat conduction (built)

- **Data:** parametric FEniCS solutions of ρcₚ ∂T/∂t = ∇·(k∇T) + Q on the unit square, with
  conductivity, heat source, initial condition and boundary configuration varied by Latin
  Hypercube Sampling. The solver is checked against exact solutions and an energy balance
  (`data/raw/heat_conduction/validate_solver.py`).
- **Model:** a 3D Fourier Neural Operator (FNO3d) that maps the parameters, initial condition and
  boundary configuration to the full T(x, y, t) field. Trained on a free Google Colab GPU, with
  the whole dataset kept on the GPU and batches built on-device.
- **Result:** **6.2 % relative L2 error** on unseen test cases. The largest errors are at sharp
  boundary gradients.
- **Deployment:** an inference helper runs the surrogate from parameters alone (no solver), and
  the notebook exports TorchScript and ONNX models.

| File | Contents |
|---|---|
| `data/raw/heat_conduction/generate.py` | Steady-state dataset generator |
| `data/raw/heat_conduction/generate_transient.py` | Transient dataset generator |
| `data/raw/heat_conduction/validate_solver.py` | Solver verification |
| `notebooks/02_transient_heat_conduction_fno.ipynb` | FNO3d training, test evaluation, ONNX export |

## Planned experiments

| # | Experiment | Solver | Surrogate |
|---|---|---|---|
| 2 | 2D flow over a bluff body | OpenFOAM | FNO |
| 3 | Fin array heat transfer | analytical | DNN |
| 4 | Beam bending and stress | FEniCS | DNN |
| 5 | Internal pipe flow | OpenFOAM | FNO |

The FastAPI backend and React front end are designed but not yet in this repository.

## Run

```bash
python3 data/raw/heat_conduction/generate_transient.py   # needs FEniCS 2019.1.0
```

Then open the notebook in Google Colab with a GPU runtime.

## Author

**Oluwatobi Ademola** · [github.com/tobi-ademola](https://github.com/tobi-ademola) · MIT License
