// Reward Schedule Editor

const ScheduleEditor = (() => {
  let container;
  let state = {
    means: deepClone(DEFAULT_STATE_MEANS),
    boundaries: deepClone(DEFAULT_BOUNDARIES),
    rewards: [],
  };

  function init(containerEl) {
    container = containerEl;
    regenerateRewards();
    render();
  }

  function regenerateRewards() {
    state.rewards = generateRewards(state.means, state.boundaries);
  }

  function render() {
    clearEl(container);

    // Version preset selector
    const presetRow = el('div', { className: 'schedule-presets' });
    presetRow.appendChild(el('span', { className: 'label' }, 'Presets: '));
    for (const [key, ver] of Object.entries(SCHEDULE_VERSIONS)) {
      const btn = el('button', { className: 'btn btn-sm', onClick: () => {
        state.boundaries = deepClone(ver.boundaries);
        regenerateRewards();
        render();
      }}, ver.label);
      presetRow.appendChild(btn);
    }
    presetRow.appendChild(el('button', { className: 'btn btn-sm btn-accent', onClick: () => {
      regenerateRewards();
      render();
    }}, '↻ Resample'));
    container.appendChild(presetRow);

    // State means table
    const meansTable = el('div', { className: 'means-table' });
    meansTable.appendChild(el('span', { className: 'label' }, 'State means: '));
    for (let i = 0; i < 3; i++) {
      const inp = el('input', {
        type: 'number',
        className: 'mean-input',
        value: state.means[i],
      });
      inp.style.borderColor = ['#93c5fd', '#fcd34d', '#f87171'][i];
      inp.addEventListener('change', e => {
        state.means[i] = parseInt(e.target.value) || 0;
        regenerateRewards();
        render();
      });
      meansTable.appendChild(el('span', { className: 'mean-label' }, `S${i + 1}: `));
      meansTable.appendChild(inp);
    }
    container.appendChild(meansTable);

    // Slot strip
    const strip = el('div', { className: 'schedule-strip' });
    for (let i = 1; i <= BLOCK_LENGTH; i++) {
      const stateIdx = getStateForPosition(i, state.boundaries);
      const isBoundary = state.boundaries.includes(i);
      const slot = el('div', {
        className: 'schedule-slot' + (isBoundary ? ' slot-boundary' : ''),
      });
      slot.style.backgroundColor = ['#dbeafe', '#fef9c3', '#fecaca'][stateIdx];
      slot.style.borderColor = ['#93c5fd', '#fcd34d', '#f87171'][stateIdx];

      const numLabel = el('div', { className: 'slot-number' }, String(i));
      const rewardLabel = el('div', { className: 'slot-reward' }, String(state.rewards[i - 1]));
      slot.appendChild(numLabel);
      slot.appendChild(rewardLabel);

      if (isBoundary) {
        slot.appendChild(el('div', { className: 'slot-boundary-tag' }, 'B'));
      }

      // Draggable boundary
      slot.draggable = true;
      slot.addEventListener('dragstart', e => {
        if (!isBoundary) { e.preventDefault(); return; }
        e.dataTransfer.setData('text/plain', String(i));
        e.dataTransfer.effectAllowed = 'move';
        slot.classList.add('dragging');
      });
      slot.addEventListener('dragend', () => slot.classList.remove('dragging'));
      slot.addEventListener('dragover', e => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        slot.classList.add('drag-over');
      });
      slot.addEventListener('dragleave', () => slot.classList.remove('drag-over'));
      slot.addEventListener('drop', e => {
        e.preventDefault();
        slot.classList.remove('drag-over');
        const fromPos = parseInt(e.dataTransfer.getData('text/plain'));
        const toPos = i;
        if (toPos < 2 || toPos > 26) return;
        const bIdx = state.boundaries.indexOf(fromPos);
        if (bIdx === -1) return;
        // Ensure boundaries stay ordered and don't overlap
        const newBounds = [...state.boundaries];
        newBounds[bIdx] = toPos;
        newBounds.sort((a, b) => a - b);
        if (newBounds[0] >= newBounds[1]) return;
        if (newBounds[0] < 2 || newBounds[1] < newBounds[0] + 2) return;
        state.boundaries = newBounds;
        regenerateRewards();
        render();
      });

      strip.appendChild(slot);
    }
    container.appendChild(strip);

    // Legend
    const legend = el('div', { className: 'schedule-legend' });
    legend.appendChild(el('span', {}, `Boundaries at positions ${state.boundaries[0]} and ${state.boundaries[1]}. `));
    legend.appendChild(el('span', {}, `State lengths: ${state.boundaries[0] - 1}, ${state.boundaries[1] - state.boundaries[0]}, ${BLOCK_LENGTH - state.boundaries[1] + 1}. `));
    legend.appendChild(el('span', { className: 'hint' }, 'Drag boundary slots (B) to reposition.'));
    container.appendChild(legend);
  }

  return { init };
})();
