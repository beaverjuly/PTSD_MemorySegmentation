// Encoding & Memory Test Preview renderer

const Preview = (() => {
  let container;
  let block3State = {
    step: 'A',
    predecessorTop: Math.random() > 0.5,
    sliderValue: 50,
  };

  function init(containerEl) {
    container = containerEl;
  }

  function resetBlock3State() {
    block3State = {
      step: 'A',
      predecessorTop: Math.random() > 0.5,
      sliderValue: 50,
    };
  }

  function renderEncodingScreen(phase, block = 1) {
    clearEl(container);
    const tint = BLOCK_TINTS[block] || BLOCK_TINTS[1];

    const wrap = el('div', {
      className: 'preview-screen encoding-screen',
      style: { backgroundColor: tint.bg },
    });

    const card = createPlaceholderCard(10, block);
    card.classList.add('card-large');
    wrap.appendChild(card);

    // Always show the reward box; leave empty in Phase A
    const reward = el('div', { className: 'reward-label reward-label-fixed' }, phase === 'B' ? '60 pts' : '');
    wrap.appendChild(reward);

    wrap.appendChild(
      el(
        'div',
        { className: 'phase-label padded-text' },
        phase === 'A' ? 'Phase A: Image only (1 s)' : 'Phase B: Image + Reward (2 s)'
      )
    );

    container.appendChild(wrap);
  }

  function renderRecognitionScreen(block) {
    clearEl(container);
    const tint = BLOCK_TINTS[block] || BLOCK_TINTS[1];

    const wrap = el('div', {
      className: 'preview-screen recognition-screen',
      style: { backgroundColor: tint.bg },
    });

    const card = createPlaceholderCard(block === 1 ? 9 : 10, block);
    card.classList.add('card-large');
    wrap.appendChild(card);

    wrap.appendChild(
      el('div', { className: 'question-text padded-text' }, 'Was this image shown earlier in this room?')
    );
    wrap.appendChild(
      el('div', { className: 'response-reminder padded-text' }, 'Press O for old and N for new.')
    );

    container.appendChild(wrap);
  }

  function renderBlock3(resetState) {
    if (resetState) resetBlock3State();

    clearEl(container);

    const wrap = el('div', {
      className: 'preview-screen block3-screen',
      style: { backgroundColor: BLOCK_TINTS[3].bg },
    });

    if (block3State.step === 'A') {
      renderBlock3StepA(wrap);
    } else {
      renderBlock3StepB(wrap);
    }

    container.appendChild(wrap);
  }

  function renderBlock3StepA(wrap) {
    wrap.appendChild(
      el('div', { className: 'step-label padded-text prominent-text' }, 'Which came first?')
    );

    const pairWrap = el('div', { className: 'b3-pair-vertical' });
    const topPos = block3State.predecessorTop ? 9 : 11;
    const botPos = block3State.predecessorTop ? 11 : 9;

    const topCard = createPlaceholderCard(topPos, 3);
    const botCard = createPlaceholderCard(botPos, 3);
    topCard.classList.add('card-medium');
    botCard.classList.add('card-medium');

    pairWrap.appendChild(topCard);
    pairWrap.appendChild(botCard);
    wrap.appendChild(pairWrap);

    [topCard, botCard].forEach(card => {
      card.style.cursor = 'pointer';
      card.addEventListener('click', () => {
        block3State.chosenFirst = parseInt(card.dataset.pos, 10);
        block3State.step = 'B';
        renderBlock3(false);
      });
    });
  }

  function renderBlock3StepB(wrap) {
    wrap.appendChild(
      el('div', { className: 'step-label padded-text prominent-text' }, 'Where did the middle image occur?')
    );

    const firstPos = block3State.chosenFirst;
    const secondPos = firstPos === 9 ? 11 : 9;

    const boundaryCardWrap = el('div', { className: 'b3-boundary-card-wrap' });
    const boundaryCard = createPlaceholderCard(10, 3);
    boundaryCard.classList.add('card-small');
    boundaryCardWrap.appendChild(boundaryCard);
    wrap.appendChild(boundaryCardWrap);

    const lineShell = el('div', { className: 'b3-line-shell slider-shell' });
    const line = el('div', { className: 'b3-line slider-line' });

    const leftCard = createPlaceholderCard(firstPos, 3);
    leftCard.classList.add('card-small', 'card-locked', 'b3-anchor-card', 'left-anchor');
    line.appendChild(leftCard);

    const rightCard = createPlaceholderCard(secondPos, 3);
    rightCard.classList.add('card-small', 'card-locked', 'b3-anchor-card', 'right-anchor');
    line.appendChild(rightCard);

    const sliderWrap = el('div', { className: 'b3-slider-wrap' });
    const slider = el('input', {
      className: 'b3-slider',
      type: 'range',
      min: '0',
      max: '100',
      step: '1',
      value: String(block3State.sliderValue),
    });

    slider.addEventListener('input', e => {
      block3State.sliderValue = parseInt(e.target.value, 10);
      updateSliderThumb(slider);
    });

    sliderWrap.appendChild(slider);
    line.appendChild(sliderWrap);

    lineShell.appendChild(line);
    wrap.appendChild(lineShell);

    wrap.appendChild(
      el(
        'div',
        { className: 'instruction-text padded-text prominent-subtext' },
        'This image appeared sometime between these two images. Move the slider to show where you think it occurred.'
      )
    );

    updateSliderThumb(slider);
  }

  function updateSliderThumb(sliderEl) {
    const value = parseInt(sliderEl.value, 10);
    sliderEl.style.setProperty('--slider-value', `${value}%`);
  }

  return { init, renderEncodingScreen, renderRecognitionScreen, renderBlock3, resetBlock3State };
})();