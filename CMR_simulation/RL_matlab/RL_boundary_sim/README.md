# Exp1 RL–Memory Simulation

Simulation framework for comparing three hypotheses about how reinforcement-learning
signals modulate encoding drift and subsequent recall at event boundaries.

## Hypotheses

- **H0 (Baseline):** Normal encoding-drift increase at boundaries, modulated by outcome RPE.
- **H1 (Smaller boundary increase):** Reduced boundary-specific drift increase for a given RPE magnitude.
- **H2 (Global bias reduction):** Same boundary-specific drift increase, but a global downward shift in encoding drift everywhere.

## Repository layout

```
repo/
├── notebooks/
│   └── exp1_RL_sim.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── task_design.py
│   ├── hypotheses.py
│   ├── simulator.py
│   ├── metrics.py
│   ├── plotting.py
│   └── utils.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── serialized/
├── data/
│   ├── raw/
│   ├── interim/
│   └── simulated/
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/exp1_RL_sim.ipynb
```

Or run from the command line:

```bash
cd repo
python -c "from src import config, simulator, metrics, plotting; exec(open('notebooks/run_all.py').read())"
```

## Design principles

- `task_design.py` decides what happened (trial structure).
- `hypotheses.py` decides how encoding drift is assigned (the experimental manipulation).
- `simulator.py` decides how latent states generate recall.
- `metrics.py` decides how you evaluate the resulting behavior.
- `plotting.py` decides how you visualize the comparison.

The task schedule is held fixed; only the encoding-drift rule changes across hypotheses;
all output metrics are computed identically.
