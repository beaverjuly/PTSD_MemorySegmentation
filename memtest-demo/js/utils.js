// Shared utility functions

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'className') e.className = v;
      else if (k === 'textContent') e.textContent = v;
      else if (k === 'innerHTML') e.innerHTML = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
      else e.setAttribute(k, v);
    }
  }
  children.forEach(c => {
    if (typeof c === 'string') e.appendChild(document.createTextNode(c));
    else if (c) e.appendChild(c);
  });
  return e;
}

function clearEl(container) {
  container.innerHTML = '';
}

function countTokenTypes(stream) {
  const counts = { critical: 0, within: 0, single: 0, new_item: 0 };
  stream.forEach(t => {
    const def = TOKEN_DEFS[t];
    if (def && counts.hasOwnProperty(def.category)) {
      counts[def.category]++;
    }
  });
  return counts;
}

function validateStream(stream) {
  const warnings = [];
  const counts = countTokenTypes(stream);

  // Minimum composition checks for the current pilot design:
  // 4 critical pairs, 2 within-state pairs, 2 single-old items, 10 new items
  if (counts.critical < 4) warnings.push('Fewer than 4 critical pairs.');
  if (counts.within < 2) warnings.push('Fewer than 2 within-state filler pairs.');
  if (counts.single < 2) warnings.push('Fewer than 2 single old items.');
  if (counts.new_item < 10) warnings.push('Fewer than 10 new items.');

  // Spacing checks for critical pairs
  for (let i = 0; i < stream.length; i++) {
    const def = TOKEN_DEFS[stream[i]];
    if (!def || def.category !== 'critical') continue;

    const boundaryNum = getBoundaryNum(stream[i]);
    const prime = def.items[0];

    // Same-boundary critical pairs should have at least 3 events in between
    for (let j = i + 1; j < Math.min(i + 4, stream.length); j++) {
      const def2 = TOKEN_DEFS[stream[j]];
      if (def2 && def2.category === 'critical' && getBoundaryNum(stream[j]) === boundaryNum) {
        warnings.push(
          `Same-boundary critical pairs "${stream[i]}" and "${stream[j]}" are too close. Need at least 3 events in between.`
        );
      }
    }

    // Same critical prime should have at least 4 events in between
    for (let j = i + 1; j < Math.min(i + 5, stream.length); j++) {
      const def2 = TOKEN_DEFS[stream[j]];
      if (def2 && def2.category === 'critical' && def2.items[0] === prime) {
        warnings.push(
          `Critical prime item ${prime} is reused too quickly between "${stream[i]}" and "${stream[j]}". Need at least 4 events in between.`
        );
      }
    }
  }

  // Soft duplicate-use check:
  // repeated use is expected for some critical items, but excessive reuse is likely accidental.
  const oldItemCounts = {};
  stream.forEach(tokenKey => {
    const def = TOKEN_DEFS[tokenKey];
    if (!def || !def.items || !def.items.length) return;

    def.items.forEach(pos => {
      oldItemCounts[pos] = (oldItemCounts[pos] || 0) + 1;
    });
  });

  Object.entries(oldItemCounts).forEach(([pos, count]) => {
    if (count > 2) {
      warnings.push(
        `Old serial-position item ${pos} appears ${count} times in this stream. Check whether this reuse is intended.`
      );
    }
  });

  return dedupeWarnings(warnings);
}

function dedupeWarnings(warnings) {
  return [...new Set(warnings)];
}

function getBoundaryNum(tokenKey) {
  if (tokenKey.includes('1')) return 1;
  if (tokenKey.includes('2')) return 2;
  return 0;
}

function tokenColor(tokenKey) {
  const def = TOKEN_DEFS[tokenKey];
  if (!def) return '#ccc';
  return TOKEN_CATEGORIES[def.category].color;
}

// Simple drag reorder for a list container
function enableDragReorder(container, getStream, setStream) {
  let dragIdx = null;

  container.addEventListener('dragstart', e => {
    const item = e.target.closest('.stream-token');
    if (!item) return;
    dragIdx = parseInt(item.dataset.idx, 10);
    e.dataTransfer.effectAllowed = 'move';
    item.classList.add('dragging');
  });

  container.addEventListener('dragover', e => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const item = e.target.closest('.stream-token');
    if (item) item.classList.add('drag-over');
  });

  container.addEventListener('dragleave', e => {
    const item = e.target.closest('.stream-token');
    if (item) item.classList.remove('drag-over');
  });

  container.addEventListener('drop', e => {
    e.preventDefault();
    const item = e.target.closest('.stream-token');
    if (!item || dragIdx === null) return;

    item.classList.remove('drag-over');
    const dropIdx = parseInt(item.dataset.idx, 10);
    if (dragIdx === dropIdx) return;

    const stream = getStream();
    const moved = stream.splice(dragIdx, 1)[0];
    stream.splice(dropIdx, 0, moved);
    setStream(stream);
  });

  container.addEventListener('dragend', () => {
    dragIdx = null;
    container.querySelectorAll('.dragging, .drag-over').forEach(el => {
      el.classList.remove('dragging', 'drag-over');
    });
  });
}