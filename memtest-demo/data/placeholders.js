// Placeholder stimulus card definitions
const BLOCK_TINTS = {
  1: { bg: '#dbeafe', border: '#93c5fd', label: 'Block 1' },
  2: { bg: '#fef9c3', border: '#fcd34d', label: 'Block 2' },
  3: { bg: '#fecaca', border: '#f87171', label: 'Block 3' },
};

function createPlaceholderCard(serialPos, block, opts = {}) {
  const tint = BLOCK_TINTS[block] || BLOCK_TINTS[1];
  const el = document.createElement('div');
  el.className = 'placeholder-card' + (opts.extraClass ? ' ' + opts.extraClass : '');

  // Neutral card fill; tint only the border
  el.style.backgroundColor = '#f8fafc';
  el.style.borderColor = tint.border;

  el.textContent = String(serialPos).padStart(2, '0');

  if (opts.small) el.classList.add('card-small');
  if (opts.draggable) {
    el.setAttribute('draggable', 'true');
    el.classList.add('card-draggable');
  }

  el.dataset.pos = serialPos;
  el.dataset.block = block;
  return el;
}
