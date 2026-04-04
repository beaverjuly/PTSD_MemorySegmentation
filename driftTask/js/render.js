/**
 * render.js – Screen renderer.
 * Draws the current state into the DOM. Does not decide task order.
 *
 * Revised so the encoding bar stays in a fixed slot across:
 * - item-only phase
 * - item+value phase
 * - encoding ITI
 *
 * Also revised so:
 * - a persistent white item frame remains on screen
 * - item images fade in briefly
 */

const Render = (() => {
  /* ---- DOM refs ---- */
  let $instruction, $encodingFrame, $itemArea, $valueArea, $barArea, $responseArea, $btnArea;

  function init() {
    $instruction   = document.getElementById('instruction-region');
    $encodingFrame = document.getElementById('encoding-frame');
    $itemArea      = document.getElementById('item-region');
    $valueArea     = document.getElementById('value-region');
    $barArea       = document.getElementById('bar-region');
    $responseArea  = document.getElementById('response-region');
    $btnArea       = document.getElementById('btn-region');
  }

  /* ===========================================================
   * VISIBILITY / CLEAR HELPERS
   * =========================================================== */

  function clearAll() {
    [
      $instruction,
      $itemArea,
      $valueArea,
      $barArea,
      $responseArea,
      $btnArea
    ].forEach(el => {
      if (el) el.innerHTML = '';
    });
  }

  function hideAll() {
    [
      $instruction,
      $encodingFrame,
      $responseArea,
      $btnArea
    ].forEach(el => {
      if (el) el.style.display = 'none';
    });
  }

  function showInstructionLayout() {
    hideAll();
    if ($instruction) $instruction.style.display = '';
    if ($btnArea) $btnArea.style.display = '';
  }

  function showEncodingLayout() {
    hideAll();
    if ($encodingFrame) $encodingFrame.style.display = '';
  }

  function showResponseLayout() {
    hideAll();
    if ($encodingFrame) $encodingFrame.style.display = '';
    if ($responseArea) $responseArea.style.display = '';
    if ($btnArea) $btnArea.style.display = '';
  }

  /* ===========================================================
   * INSTRUCTION SCREEN
   * =========================================================== */

  function showInstruction(text, onNext) {
    clearAll();
    showInstructionLayout();

    $instruction.innerHTML = `
      <div class="instr-text">${String(text).replace(/\n/g, '<br>')}</div>
    `;

    if (typeof onNext === 'function') {
      const btn = document.createElement('button');
      btn.className = 'btn-next';
      btn.textContent = 'NEXT';
      btn.onclick = onNext;
      $btnArea.appendChild(btn);
    }
  }

  /* ===========================================================
   * ENCODING SCREENS
   * =========================================================== */

  function showItemOnly(filename, trialIndex) {
    clearAll();
    showEncodingLayout();

    renderItemImage(filename, true);   // fade only on first appearance
    renderHiddenValuePlaceholder();
    renderBar(trialIndex);
  }

  function showItemValue(filename, value, trialIndex) {
    showEncodingLayout();

    // Do NOT re-render the image if it is already there
    const existingImg = $itemArea.querySelector('.stim-img');
    if (!existingImg) {
      renderItemImage(filename, false);
    }

    renderValue(value);
    renderBar(trialIndex);
  }

  function showEncodingITI(trialIndex) {
    clearAll();
    showEncodingLayout();

    renderEmptyItemFrame();
    renderHiddenValuePlaceholder();
    renderBar(trialIndex);
  }

  function showFixation(trialIndex = null) {
    clearAll();
    showEncodingLayout();

    renderFixationInFrame();
    renderHiddenValuePlaceholder();

    if (trialIndex !== null && trialIndex !== undefined) {
      renderBar(trialIndex);
    } else {
      renderEmptyBarSlot();
    }
  }

  function showBlank() {
    clearAll();
    hideAll();
  }

  /* ===========================================================
   * SINGLE-ITEM PLACEMENT TEST
   * =========================================================== */

  function showSinglePlacement(filename, totalTrials, onResponse) {
    clearAll();
    showResponseLayout();

    renderItemImage(filename);
    renderEmptyValueSlot();
    renderEmptyBarSlot();

    renderPlacementScale(totalTrials, (clickProportion) => {
      onResponse(clickProportion);
    });
  }

  /* ===========================================================
   * PAIRED-ITEM CO-PLACEMENT TEST
   * =========================================================== */

  function showPairedPlacement(filenameA, filenameB, totalTrials, onResponse) {
    clearAll();
    showResponseLayout();

    $itemArea.innerHTML = `
      <div class="paired-items">
        <div class="paired-item">
          <div class="item-frame">
            ${imgTag(filenameA, 'item-a')}
          </div>
        </div>
        <div class="paired-item">
          <div class="item-frame">
            ${imgTag(filenameB, 'item-b')}
          </div>
        </div>
      </div>
    `;

    renderEmptyValueSlot();
    renderEmptyBarSlot();

    renderPairedPlacementScale(totalTrials, onResponse);
  }

  /* ===========================================================
   * RECOGNITION TEST
   * =========================================================== */

  function showRecognition(filename, onResponse) {
    clearAll();
    showResponseLayout();

    renderItemImage(filename);
    renderEmptyValueSlot();
    renderEmptyBarSlot();

    const wrap = document.createElement('div');
    wrap.className = 'recog-buttons';

    const btnOld = document.createElement('button');
    btnOld.className = 'btn-recog btn-old';
    btnOld.textContent = CONFIG.keys.oldLabel;
    btnOld.onclick = () => {
      cleanupRecognitionHandler(handler);
      onResponse('old');
    };

    const btnNew = document.createElement('button');
    btnNew.className = 'btn-recog btn-new';
    btnNew.textContent = CONFIG.keys.newLabel;
    btnNew.onclick = () => {
      cleanupRecognitionHandler(handler);
      onResponse('new');
    };

    wrap.appendChild(btnOld);
    wrap.appendChild(btnNew);
    $btnArea.appendChild(wrap);

    const handler = (e) => {
      if (e.key === CONFIG.keys.oldKey) {
        cleanupRecognitionHandler(handler);
        onResponse('old');
      }
      if (e.key === CONFIG.keys.newKey) {
        cleanupRecognitionHandler(handler);
        onResponse('new');
      }
    };

    document.addEventListener('keydown', handler);
  }

  function cleanupRecognitionHandler(handler) {
    document.removeEventListener('keydown', handler);
  }

  /* ===========================================================
   * END SCREEN
   * =========================================================== */

  function showEnd(text) {
    clearAll();
    showInstructionLayout();
    $instruction.innerHTML = `
      <div class="instr-text">${String(text).replace(/\n/g, '<br>')}</div>
    `;
  }

  /* ===========================================================
   * INTERNAL HELPERS
   * =========================================================== */

  function imgTag(filename, cls = '', animate = true) {
    const src = CONFIG.imgDir + filename;
    const animClass = animate ? 'stim-fade' : '';
    return `
      <img
        class="stim-img ${cls} ${animClass}"
        src="${src}"
        alt="${filename}"
        width="${CONFIG.imgSize}"
        height="${CONFIG.imgSize}"
        onerror="this.style.background='${CONFIG.placeholderColor}'; this.alt='${filename}'"
      >
    `;
  }

  function renderItemImage(filename, animate = true) {
    $itemArea.innerHTML = `
      <div class="item-wrap">
        <div class="item-frame">
          ${imgTag(filename, '', animate)}
        </div>
      </div>
    `;
  }

  function renderEmptyItemFrame() {
    $itemArea.innerHTML = `
      <div class="item-wrap">
        <div class="item-frame item-frame-empty"></div>
      </div>
    `;
  }

  function renderFixationInFrame() {
    $itemArea.innerHTML = `
      <div class="item-wrap">
        <div class="item-frame">
          <div class="fixation">+</div>
        </div>
      </div>
    `;
  }

  function renderValue(value) {
    $valueArea.innerHTML = `
      <div class="value-wrap">
        <div class="value-frame">
          <div class="reward-value">${value}</div>
        </div>
      </div>
    `;
  }

  function renderHiddenValuePlaceholder() {
    $valueArea.innerHTML = `
      <div class="value-wrap">
        <div class="value-frame">
          <div class="reward-value reward-hidden">&nbsp;</div>
        </div>
      </div>
    `;
  }

  function renderEmptyValueSlot() {
    $valueArea.innerHTML = `
      <div class="value-wrap">
        <div class="value-frame">
          <div class="reward-value reward-hidden">&nbsp;</div>
        </div>
      </div>
    `;
  }

  function renderEmptyBarSlot() {
    $barArea.innerHTML = `<div class="bar-placeholder"></div>`;
  }

  /**
   * Render the encoding bar.
   * In simple version, reserve the slot with an invisible placeholder.
   * In rich version, show the actual bar in the same fixed region.
   */
  function renderBar(trialIndex) {
    if (!CONFIG.bar.show) {
      renderEmptyBarSlot();
      return;
    }

    const fill = trialToBarFill(trialIndex, CONFIG.trialsPerBlock);
    const grad = barGradient(fill);

    $barArea.innerHTML = `
      <div
        class="bar-container"
        style="
          width:${CONFIG.bar.width};
          height:${CONFIG.bar.height}px;
          border:1px solid ${CONFIG.bar.borderColor};
          background:${CONFIG.bar.emptyColor};
        "
      >
        <div
          class="bar-fill"
          style="
            width:${(fill * 100).toFixed(1)}%;
            background:${grad};
          "
        ></div>
      </div>
    `;
  }

  /**
   * Single-item placement scale.
   */
  function renderPlacementScale(totalTrials, onClick) {
    const isRich = CONFIG.bar.show;

    const scaleEl = document.createElement('div');
    scaleEl.className = 'placement-scale' + (isRich ? ' scale-rich' : ' scale-plain');
    scaleEl.style.width = CONFIG.scale.width;
    scaleEl.style.height = `${CONFIG.scale.height}px`;

    if (isRich) {
      scaleEl.style.background = barGradient(1.0);
      scaleEl.style.border = `1px solid ${CONFIG.bar.borderColor}`;
      scaleEl.style.borderRadius = '4px';
    } else {
      scaleEl.style.background = 'transparent';
      scaleEl.style.borderBottom = `3px solid ${CONFIG.scale.plainColor}`;
    }

    const labelWrap = document.createElement('div');
    labelWrap.className = 'scale-labels';
    labelWrap.innerHTML = '<span>Start</span><span>End</span>';

    let marker = null;
    let selectedProp = null;

    scaleEl.addEventListener('click', (e) => {
      const rect = scaleEl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const proportion = clamp(x / rect.width, 0, 1);
      selectedProp = proportion;

      if (marker) marker.remove();

      marker = document.createElement('div');
      marker.className = 'scale-marker';
      marker.style.left = `${proportion * 100}%`;
      scaleEl.appendChild(marker);

      let confirmBtn = $responseArea.querySelector('.btn-confirm');
      if (!confirmBtn) {
        confirmBtn = document.createElement('button');
        confirmBtn.className = 'btn-confirm';
        confirmBtn.textContent = 'Confirm';
        confirmBtn.onclick = () => onClick(selectedProp);
        $responseArea.appendChild(confirmBtn);
      }
    });

    $responseArea.appendChild(scaleEl);
    $responseArea.appendChild(labelWrap);
  }

  /**
   * Paired placement scale.
   */
  function renderPairedPlacementScale(totalTrials, onResponse) {
    const isRich = CONFIG.bar.show;

    const scaleEl = document.createElement('div');
    scaleEl.className = 'placement-scale paired-scale' + (isRich ? ' scale-rich' : ' scale-plain');
    scaleEl.style.width = CONFIG.scale.width;
    scaleEl.style.height = `${CONFIG.scale.height}px`;

    if (isRich) {
      scaleEl.style.background = barGradient(1.0);
      scaleEl.style.border = `1px solid ${CONFIG.bar.borderColor}`;
      scaleEl.style.borderRadius = '4px';
    } else {
      scaleEl.style.background = 'transparent';
      scaleEl.style.borderBottom = `3px solid ${CONFIG.scale.plainColor}`;
    }

    const labelWrap = document.createElement('div');
    labelWrap.className = 'scale-labels';
    labelWrap.innerHTML = '<span>Start</span><span>End</span>';

    let markerA = null;
    let markerB = null;
    let posA = null;
    let posB = null;
    let placingA = true;

    const placingLabel = document.createElement('div');
    placingLabel.className = 'placing-label';
    placingLabel.textContent = 'Click to place Image A';

    scaleEl.addEventListener('click', (e) => {
      const rect = scaleEl.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const proportion = clamp(x / rect.width, 0, 1);

      if (placingA) {
        if (markerA) markerA.remove();

        markerA = document.createElement('div');
        markerA.className = 'scale-marker marker-a';
        markerA.style.left = `${proportion * 100}%`;
        markerA.textContent = 'A';
        scaleEl.appendChild(markerA);

        posA = proportion;
        placingA = false;
        placingLabel.textContent = 'Click to place Image B';
      } else {
        if (markerB) markerB.remove();

        markerB = document.createElement('div');
        markerB.className = 'scale-marker marker-b';
        markerB.style.left = `${proportion * 100}%`;
        markerB.textContent = 'B';
        scaleEl.appendChild(markerB);

        posB = proportion;
        placingLabel.textContent = 'Both placed. Confirm or click to adjust.';

        let confirmBtn = $responseArea.querySelector('.btn-confirm');
        if (!confirmBtn) {
          confirmBtn = document.createElement('button');
          confirmBtn.className = 'btn-confirm';
          confirmBtn.textContent = 'Confirm';
          confirmBtn.onclick = () => onResponse({ posA, posB });
          $responseArea.appendChild(confirmBtn);
        }

        placingA = true;
      }
    });

    $responseArea.appendChild(placingLabel);
    $responseArea.appendChild(scaleEl);
    $responseArea.appendChild(labelWrap);
  }

  /* ===========================================================
   * PUBLIC API
   * =========================================================== */

  return {
    init,
    clearAll,
    showInstruction,
    showItemOnly,
    showItemValue,
    showEncodingITI,
    showFixation,
    showBlank,
    showSinglePlacement,
    showPairedPlacement,
    showRecognition,
    showEnd
  };
})();
