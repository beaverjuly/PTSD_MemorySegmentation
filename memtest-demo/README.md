# MemTest Demo

Static HTML/CSS/JS demo for the memory-test experiment design.

## Sections

1. **Encoding & Memory Test Preview** – participant-facing screen mockups for encoding trials (Phase A image-only, Phase B image+reward), Block 1 & 2 recognition, and Block 3 temporal-order/boundary-distance task.

2. **Reward Schedule Editor** – 27-slot strip showing 3 latent states. Drag boundary markers to reposition. Edit state means; rewards auto-regenerate. Preset schedule versions A/B/C.

3. **Recognition Stream Builder** – drag-reorder, duplicate, delete tokens for Block 1 and Block 2 recognition streams. Color-coded by category. Automatic validation warnings for spacing violations and minimum counts.

## Token Labels

- `B1→P1` = Boundary 1 → Predecessor 1
- `P1→S1` = Predecessor 1 → Successor 1 (across boundary)
- `W1` = Within-State Filler Pair 1
- `OOS1` = Out-of-Sequence Old–Old Baseline 1
- `SO1` = Single Old Item 1
- `New` = New Item

## Data Files

- `data/schedules.js` – default boundary positions, state means, reward generation
- `data/streams.js` – default Block 1 & 2 streams, token definitions, categories
- `data/placeholders.js` – CSS placeholder card generator with block tints

## Usage

Open `index.html` in any browser. No build step required.
