# Prototype: Reward / Loss Task Demos

A **100% standalone** sandbox for demoing the visual reskin, feedback logic, and one mock trial — with zero dependencies on Flask, jsPsych, or the real experiment backend.

Open any `.html` file directly in a browser.

---

## Files

### `demo_visual.html`
**What it tests:** Art direction for reward vs loss contexts.

| Element | Reward | Loss |
|---|---|---|
| Drone (was: bird) | Green body, green propellers | Red body, red propellers |
| Drop bag | Gold supply pod | Dark red hazard pod |
| Falling objects | Gold coins | Red hazard droplets |
| Bucket / Shield | Red collector | Blue shield |
| Sky | Warm sunset gradient | Dark violet gradient |
| Land | Green terrain | Dark grey terrain |

**Corresponds to:** `game.min.css` (sprite classes), `trial.js` (`make_html()`), `static/img/*`

**Toggle** the Reward / Loss button to compare both contexts side-by-side in the same layout.

---

### `demo_feedback.html`
**What it tests:** Accuracy → value mapping and feedback color rules.

Four modes:

| Mode | Formula | Example (caught 7) |
|---|---|---|
| Original | value = captureCount | 7 |
| Reward | value = captureCount | +7 (green) |
| Loss | value = captureCount − 10 | −3 (red) |
| Mixed | value = captureCount − 5 | +2 (green) |

Color rules:
- Positive → green (#39FF14)
- Negative → red (#ff4444)
- Zero → yellow (#ffd700)

**Corresponds to:** `trial.js` (captureCount computation, feedback box creation, color logic)

---

### `demo_trial.html`
**What it tests:** One playable mock trial with full mechanics.

- **← →** keys move the collector/shield
- Click **Drop Reward** or **Drop Hazard** to run a trial
- Drone appears, bag drops, objects scatter, capture is computed via pixel overlap
- Feedback shows mapped value with sign and color
- A stimulus placeholder flashes after the drop (representing the memory item)

Uses real values from `stimuli.js` (bag position 64.156, bird position 60) and `stimuli-details.js` (coin spread and timing arrays).

**Corresponds to:** `trial.js` (movement, `fly()`, capture, feedback), `game.min.css` (layout/animation), `stimuli-details.js` (spread/timing), `stimuli.js` (positions)

---

## Assets (placeholder)

All visual assets are drawn with CSS gradients and shapes — no image files are required to run the demos. When real art is ready, place files into:

```
assets/
  shared/       sky.png, layer1–4.png, expl.png, bucket1.png, bucket2.png
  reward/       drone0–4.png, supply-bag.png, supply-dot.png
  loss/         hazard-bag.png, hazard-dot.png
  items/        anchor.jpg, airplane.jpg, backpack.jpg, hammer.jpg, watch.jpg
```

---

## What is intentionally mocked

- No jsPsych timeline or plugin system
- No backend / Flask / data saving
- No full 50-trial block sequence
- No memory test
- No instruction flow
- No volatility/stochasticity scheduling
- Stimulus image shown as a labelled placeholder box

## What is intentionally preserved from the original

- Same DOM layering (sky → light → land → main-container)
- Same bucket movement step size (2%)
- Same bucket bounds (10%–90%)
- Same coin spread array from `stimuli-details.js`
- Same coin duration array
- Same capture-fraction timing (19/30)
- Same pixel-overlap capture check (100% coin width required)
- Same feedback positioning logic
- Same bag-drop animation sequence (initial delay → bag drop → explosion → coins)
