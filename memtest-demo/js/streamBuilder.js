// Recognition Stream Builder

const StreamBuilder = (() => {
  let container;
  let streams = { block1: [], block2: [] };

  function init(containerEl) {
    container = containerEl;
    resetToDefaults();
  }

  function resetToDefaults() {
    streams.block1 = deepClone(DEFAULT_BLOCK1_STREAM);
    streams.block2 = deepClone(DEFAULT_BLOCK2_STREAM);
    render();
  }

  function render() {
    clearEl(container);

    const resetRow = el('div', { className: 'stream-reset-row' });
    resetRow.appendChild(
      el('button', { className: 'btn btn-sm btn-accent', onClick: resetToDefaults }, '↻ Reset to Defaults')
    );
    container.appendChild(resetRow);

    const panels = el('div', { className: 'stream-panels' });
    panels.appendChild(
      renderStreamPanel('Block 1', streams.block1, BLOCK1_TOKENS, s => {
        streams.block1 = s;
        render();
      })
    );
    panels.appendChild(
      renderStreamPanel('Block 2', streams.block2, BLOCK2_TOKENS, s => {
        streams.block2 = s;
        render();
      })
    );
    container.appendChild(panels);
  }

  function renderStreamPanel(title, stream, availableTokens, setStream) {
    const panel = el('div', { className: 'stream-panel' });
    panel.appendChild(el('div', { className: 'stream-panel-title' }, title));

    const addRow = el('div', { className: 'stream-add-row' });
    const sel = el('select', { className: 'token-select' });
    availableTokens.forEach(t => {
      sel.appendChild(el('option', { value: t, textContent: TOKEN_DEFS[t].label }));
    });
    addRow.appendChild(sel);
    addRow.appendChild(
      el(
        'button',
        {
          className: 'btn btn-sm',
          onClick: () => {
            stream.push(sel.value);
            setStream(stream);
          },
        },
        '+ Add'
      )
    );
    panel.appendChild(addRow);

    const list = el('div', { className: 'stream-list' });
    stream.forEach((tok, idx) => {
      const def = TOKEN_DEFS[tok];
      const item = el('div', {
        className: 'stream-token',
        draggable: 'true',
        title: def ? def.expanded : tok,
      });
      item.dataset.idx = idx;
      item.style.borderLeftColor = tokenColor(tok);

      const num = el('span', { className: 'token-num' }, String(idx + 1));
      const label = el('span', { className: 'token-label' }, def ? def.label : tok);
      const desc = el('span', { className: 'token-items' }, def ? def.expanded : '');

      const actions = el('span', { className: 'token-actions' });
      actions.appendChild(
        el(
          'button',
          {
            className: 'btn-icon',
            title: 'Duplicate',
            onClick: e => {
              e.stopPropagation();
              stream.splice(idx + 1, 0, tok);
              setStream(stream);
            },
          },
          '⧉'
        )
      );
      actions.appendChild(
        el(
          'button',
          {
            className: 'btn-icon btn-del',
            title: 'Delete',
            onClick: e => {
              e.stopPropagation();
              stream.splice(idx, 1);
              setStream(stream);
            },
          },
          '✕'
        )
      );

      item.appendChild(num);
      item.appendChild(label);
      item.appendChild(desc);
      item.appendChild(actions);
      list.appendChild(item);
    });

    enableDragReorder(list, () => stream, s => {
      stream.length = 0;
      stream.push(...s);
      setStream(stream);
    });
    panel.appendChild(list);

    const warnings = validateStream(stream);
    if (warnings.length) {
      const warnBox = el('div', { className: 'validation-warnings' });
      warnBox.appendChild(el('div', { className: 'warn-title' }, '⚠ Validation Warnings'));
      warnings.forEach(w => warnBox.appendChild(el('div', { className: 'warn-item' }, w)));
      panel.appendChild(warnBox);
    } else {
      panel.appendChild(el('div', { className: 'validation-ok' }, '✓ Stream passes all checks'));
    }

    panel.appendChild(renderSummaryPanel(stream));
    return panel;
  }

  function renderSummaryPanel(stream) {
    const counts = summarizeStream(stream);
    const wrap = el('div', { className: 'stream-summary-panel' });

    wrap.appendChild(el('div', { className: 'summary-header' }, 'Event Summary'));

    const list = el('div', { className: 'summary-list' });

    const totalRow = summaryRow('Total Sequence Length', counts.total);
    totalRow.classList.add('summary-row-total');
    list.appendChild(totalRow);

    list.appendChild(el('hr', { className: 'summary-divider' }));

    list.appendChild(summaryRow('Critical Pairs', counts.critical));
    list.appendChild(summaryRow('Within-state Fillers', counts.within));
    list.appendChild(summaryRow('Unused Old Items', counts.single));
    list.appendChild(summaryRow('New Items', counts.new_item));

    wrap.appendChild(list);
    return wrap;
  }

  function summaryRow(label, value) {
    const row = el('div', { className: 'summary-row' });
    row.appendChild(el('span', { className: 'summary-label' }, label));
    row.appendChild(el('span', { className: 'summary-value' }, String(value)));
    return row;
  }

  function summarizeStream(stream) {
    const counts = {
      total: stream.length,
      critical: 0,
      within: 0,
      single: 0,
      new_item: 0,
    };

    stream.forEach(tok => {
      const def = TOKEN_DEFS[tok];
      if (!def) return;
      if (counts.hasOwnProperty(def.category)) {
        counts[def.category] += 1;
      }
    });

    return counts;
  }

  return { init };
})();