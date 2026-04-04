# Temporal-Context Pilot

A browser-based pilot experiment comparing Simple vs. Rich temporal-context encoding, with within-subject boundary manipulation modeled on Rouhani (2020).

## Design overview

### Two versions (between-subjects)

- **Version A (Simple)**: Reward sequence only; placement tested on a plain horizontal line.
- **Version B (Rich)**: Same reward sequence plus a blue monotonically filling bar during encoding; placement tested on the same blue bar.

### Two encoding blocks (within-subjects)

- **No-Boundary**: 36 trials, single stable reward mean (±5 fluctuation).
- **Boundary**: 36 trials, 4 latent states × 9 trials each, 3 abrupt mean shifts (change-points at trials 9, 18, 27).

### Test types

- **Single-item placement**: Place one old item on the temporal scale.
- **Paired-item co-placement**: Place two old items simultaneously; derive order and distance.
- **Recognition**: Old/new judgment (boundary block only).

### Counterbalancing

4 groups: Version (A/B) × Block Order (no-boundary-first / boundary-first).

## How to run

### Main experiment

Open `index.html` in a browser. Use URL parameters:

```
index.html?version=simple&order=nb_first&pid=P001
index.html?version=rich&order=b_first&pid=P002
```

- `version`: `simple` | `rich`
- `order`: `nb_first` | `b_first`
- `pid`: participant ID (auto-generated if omitted)

### Demo browser

Open `demo/index.html` to visually browse all screen configurations without running the full task.

## Editing stimuli and schedules

| What to change | Where to edit |
|---|---|
| Item list | `data/items.json` |
| Reward means, change-points, test sampling | `data/schedules.json` |
| Participant-facing text | `data/instructions.json` |
| Timing, bar appearance, trial counts | `js/config.js` |
| Boundary positions | `js/config.js` → `boundary.changePoints` |
| Bar color ramp | `js/config.js` → `bar.colorStart` / `bar.colorEnd` |


## Data output

At experiment end, a JSON file is automatically downloaded containing all trial-level data (encoding, recognition, placement responses, RTs).

## Repo structure

```
temporal-context-pilot/
├── index.html          # Main experiment page
├── README.md           # This file
├── style.css           # Shared visual rules
├── js/
│   ├── config.js       # All task parameters
│   ├── taskFlow.js     # Experiment controller
│   ├── render.js       # Screen renderer
│   └── utils.js        # Helper functions
├── assets/img/         # Stimulus images (placeholders)
├── data/
│   ├── items.json      # Item registry
│   ├── schedules.json  # Reward schedules & test sampling
│   └── instructions.json # Participant-facing text
└── demo/
    ├── index.html      # Demo launcher
    ├── style.css       # Demo styling
    ├── demo.js         # Demo controller
    └── README.md       # Demo guide
```
