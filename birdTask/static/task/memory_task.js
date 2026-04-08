// memory_task.js

/*
 * jsPsych plugin for a three-stage temporal memory test.
 *
 * Stage 1: Temporal order judgement ("Which came first?")
 * Stage 2: Intervening-item count ("How many items between these two?")
 * Stage 3: Slider placement (only for 6 selected middle-item pairs:
 * 3 boundary-middle and 3 nonboundary-middle)
 * Visual style derived from preview.js Block 3 rendering pattern.
 */

jsPsych.plugins['memory-task'] = (function() {
  var plugin = {};

  plugin.info = {
    name: 'memory-task',
    description: 'Three-stage temporal order, distance, and slider placement memory test.',
    parameters: {
      pair_index: {
        type: jsPsych.plugins.parameterType.INT,
        default: undefined
      },
      block_num: {
        type: jsPsych.plugins.parameterType.INT,
        default: undefined
      },
    }
  };

  // Fixed pair list (1-based indices within a 50-trial block)
  var PREDEFINED_PAIRS = [
    [2, 4], [6, 9], [7, 10], [11, 13], [16, 18], [17, 20],
    [22, 24], [23, 26], [28, 31], [30, 33], [34, 36], [35, 38],
    [39, 41], [42, 45]
  ];

  // Six slider-eligible pairs
  var SLIDER_PAIRS_LOCAL =
    (typeof SLIDER_PAIRS !== 'undefined') ? SLIDER_PAIRS : [[2,4],[11,13],[16,18],[22,24],[34,36],[39,41]];

  var BOUNDARY_MIDDLE_PAIRS_LOCAL =
    (typeof BOUNDARY_MIDDLE_PAIRS !== 'undefined') ? BOUNDARY_MIDDLE_PAIRS : [[2,4],[11,13],[39,41]];

  var NONBOUNDARY_MIDDLE_PAIRS_LOCAL =
    (typeof NONBOUNDARY_MIDDLE_PAIRS !== 'undefined') ? NONBOUNDARY_MIDDLE_PAIRS : [[16,18],[22,24],[34,36]];

  if (typeof window !== 'undefined') {
    window.PREDEFINED_PAIRS = PREDEFINED_PAIRS;
  }

  var pairOrderByBlock = {};
  var pairProgressByBlock = {};

  function nextPairForBlock(block) {
    if (!pairOrderByBlock.hasOwnProperty(block)) {
      pairOrderByBlock[block] = jsPsych.randomization.shuffle(PREDEFINED_PAIRS.slice());
      pairProgressByBlock[block] = 0;
    }
    var order = pairOrderByBlock[block];
    var progress = pairProgressByBlock[block];
    if (progress >= order.length) return null;
    var pair = order[progress];
    pairProgressByBlock[block] = progress + 1;
    return pair;
  }

  function pairEquals(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }

  function pairInList(pair, list) {
    for (var i = 0; i < list.length; i++) {
      if (pairEquals(pair, list[i])) return true;
    }
    return false;
  }

  function isSliderPair(pair) {
    return pairInList(pair, SLIDER_PAIRS_LOCAL);
  }

  // Shared visual helpers

  function createBlock3Wrapper() {
    var wrap = document.createElement('div');
    wrap.className = 'preview-screen block3-screen';
    wrap.style.cssText =
      'display:flex;flex-direction:column;align-items:center;justify-content:center;' +
      'height:80vh;width:100%;margin:0 auto;';
    return wrap;
  }

  function createStepLabel(text) {
    var label = document.createElement('div');
    label.className = 'step-label padded-text prominent-text';
    label.style.cssText =
      'font-size:28px;text-align:center;margin-bottom:20px;min-height:42px;font-weight:bold;';
    label.textContent = text;
    return label;
  }

  function createCardImg(src, size) {
    var card = document.createElement('div');
    card.style.cssText = 'display:flex;flex-direction:column;align-items:center;';
    var img = document.createElement('img');
    img.src = src;
    var maxW = size === 'medium' ? '300px' : (size === 'small' ? '140px' : '300px');
    img.style.cssText =
      'max-width:' + maxW + ';height:auto;border:2px solid rgba(255,255,255,.18);' +
      'border-radius:8px;background:rgba(255,255,255,.08);';
    card.appendChild(img);
    return { card: card, img: img };
  }

  function createSubtext(text) {
    var p = document.createElement('p');
    p.style.cssText = 'font-size:16px;text-align:center;color:#555;margin:0;';
    p.textContent = text;
    return p;
  }

  plugin.trial = function(display_element, trial) {
    document.body.style.backgroundImage = '';
    document.body.style.backgroundSize = '';
    document.body.style.backgroundPosition = '';
    document.body.style.backgroundRepeat = '';

    var block = trial.block_num;

    function _getBlockParamsFromData() {
      try {
        var rec = jsPsych.data.get()
          .filterCustom(function(r) { return Array.isArray(r.true_vol) && r.true_vol.length; })
          .last(1).values()[0];

        if (rec && Array.isArray(rec.true_vol) && Array.isArray(rec.true_stc)) {
          return {
            true_vol: rec.true_vol,
            true_stc: rec.true_stc,
            true_valence: rec.true_valence || null
          };
        }
      } catch (e) {}
      return { true_vol: null, true_stc: null, true_valence: null };
    }

    var _bp = _getBlockParamsFromData();
    var _trueVolParam = (Array.isArray(_bp.true_vol) && _bp.true_vol.length >= block) ? _bp.true_vol[block - 1] : null;
    var _trueStcParam = (Array.isArray(_bp.true_stc) && _bp.true_stc.length >= block) ? _bp.true_stc[block - 1] : null;
    var _valence = (Array.isArray(_bp.true_valence) && _bp.true_valence.length >= block) ? _bp.true_valence[block - 1] : null;

    // Current convention:
    // vol 49 = high volatility, vol 4 = low volatility
    // stc 64 = high stochasticity, stc 16 = low stochasticity
    var _volLevel = (_trueVolParam === null) ? null : (_trueVolParam === 49 ? 'high' : 'low');
    var _stcLevel = (_trueStcParam === null) ? null : (_trueStcParam === 64 ? 'high' : 'low');
    var _condition = (_trueVolParam === null || _trueStcParam === null) ? null : ('vol' + _trueVolParam + '_stc' + _trueStcParam);

    var pair = nextPairForBlock(block);

    if (pair === null) {
      jsPsych.finishTrial({
        task_phase: 'memory',
        block: block,
        true_vol_param: _trueVolParam,
        true_stc_param: _trueStcParam,
        valence: _valence,
        vol_level: _volLevel,
        stc_level: _stcLevel,
        condition: _condition,
        skipped_pair: true,
        pair_index: trial.pair_index,
        trial1_index: null,
        trial2_index: null,
        stim_left_img: null,
        stim_right_img: null,
        stim_first_actual: null,
        order_choice_side: null,
        order_choice_img: null,
        order_correct: null,
        order_correct_bin: null,
        order_rt: null,
        distance_estimate: null,
        distance_rt: null,
        pair_true_distance: null,
        attempt_number: null,
        timed_out: null,
        placement_slider_value: null,
        placement_rt: null,
        middle_item_index: null,
        middle_item_img: null,
        middle_item_is_boundary: null,
        placement_trial_type: 'none',
        placement_error_from_true_midpoint: null
      });
      return;
    }

    var idx1 = pair[0];
    var idx2 = pair[1];
    var hasSlider = isSliderPair(pair);
    var middleItemIndex = hasSlider ? Math.floor((idx1 + idx2) / 2) : null;

    var block_trials = jsPsych.data.get()
      .filter({ trial_type: 'trial', block: block })
      .filterCustom(function(t) { return t.true_vol_param !== null && t.true_vol_param !== undefined; })
      .values();

    block_trials.sort(function(a, b) {
      return (a.trial || 0) - (b.trial || 0);
    });

    var stim1 = block_trials[idx1 - 1] ? block_trials[idx1 - 1].stim_img : undefined;
    var stim2 = block_trials[idx2 - 1] ? block_trials[idx2 - 1].stim_img : undefined;
    var middleStim = (middleItemIndex && block_trials[middleItemIndex - 1]) ? block_trials[middleItemIndex - 1].stim_img : null;

    var true_first_idx = Math.min(idx1, idx2);
    var true_first_img = (true_first_idx === idx1) ? stim1 : stim2;

    var order_images = jsPsych.randomization.shuffle([stim1, stim2]);
    var left_img = order_images[0];
    var right_img = order_images[1];

    var order_choice_side = null, order_choice_img = null, order_correct = null;
    var order_rt = null, distance_rt = null, placement_rt = null;
    var placement_slider_value = null;
    var attempt_number = 1, distance_attempt_number = 1;
    var timed_out = false, responded = false;
    var orderTimeoutID = null;
    var distancePromptShown = false;

    function renderOrderScreen() {
      responded = false;
      var wrap = createBlock3Wrapper();
      wrap.id = 'order-container';
      wrap.style.visibility = 'hidden';

      wrap.appendChild(createStepLabel('Which came first?'));

      var pairWrap = document.createElement('div');
      pairWrap.className = 'b3-pair-vertical';
      pairWrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:20px;margin-bottom:20px;';

      var top = createCardImg(left_img, 'medium');
      var topLabel = document.createElement('div');
      topLabel.style.cssText = 'font-size:24px;font-weight:bold;margin-top:10px;color:#333;';
      topLabel.textContent = '1';
      top.card.appendChild(topLabel);
      top.img.id = 'order-img-top';

      var bot = createCardImg(right_img, 'medium');
      var botLabel = document.createElement('div');
      botLabel.style.cssText = 'font-size:24px;font-weight:bold;margin-top:10px;color:#333;';
      botLabel.textContent = '2';
      bot.card.appendChild(botLabel);
      bot.img.id = 'order-img-bot';

      pairWrap.appendChild(top.card);
      pairWrap.appendChild(bot.card);
      wrap.appendChild(pairWrap);

      var instrWrap = document.createElement('div');
      instrWrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;min-height:110px;';
      instrWrap.appendChild(createSubtext('Press 1 or 2 to select.'));
      wrap.appendChild(instrWrap);

      display_element.innerHTML = '';
      display_element.appendChild(wrap);

      var start_time = performance.now();
      var loaded = 0;

      function onLoad() {
        loaded++;
        if (loaded >= 2) wrap.style.visibility = 'visible';
      }

      top.img.addEventListener('load', onLoad);
      bot.img.addEventListener('load', onLoad);
      if (top.img.complete) onLoad();
      if (bot.img.complete) onLoad();

      function keyHandler(e) {
        if (e.repeat) return;
        if (window.__memorySuppressKeysUntil && performance.now() < window.__memorySuppressKeysUntil) {
          e.preventDefault();
          return;
        }
        if (responded) return;

        if (e.key === '1' || e.key === '2') {
          e.preventDefault();
          responded = true;
          clearTimeout(orderTimeoutID);
          order_rt = Math.round(performance.now() - start_time);
          order_choice_side = e.key === '1' ? 'left' : 'right';
          order_choice_img = e.key === '1' ? left_img : right_img;
          order_correct = (order_choice_img === true_first_img);
          document.removeEventListener('keydown', keyHandler);
          renderDistancePrompt();
        }
      }

      document.addEventListener('keydown', keyHandler);

      orderTimeoutID = setTimeout(function() {
        if (!responded) {
          responded = true;
          timed_out = true;
          document.removeEventListener('keydown', keyHandler);
          wrap.style.border = '4px solid red';

          var msg = document.createElement('p');
          msg.style.cssText = 'font-size:24px;margin-top:20px;color:#d9534f;';
          msg.textContent = 'No selection made. Please make a selection within the allowed time.';
          wrap.appendChild(msg);

          setTimeout(function() {
            if (attempt_number < 2) {
              attempt_number++;
              renderOrderScreen();
            } else {
              finishTrial(null, Math.abs(idx2 - idx1) - 1, true);
            }
          }, 5000);
        }
      }, 7500);
    }

    function renderDistancePrompt(forceRerender) {
      if (distancePromptShown && forceRerender !== true) return;
      distancePromptShown = true;

      var true_distance = Math.abs(idx2 - idx1) - 1;

      var wrap = createBlock3Wrapper();
      wrap.id = 'distance-container';
      wrap.style.visibility = 'hidden';

      wrap.appendChild(createStepLabel('How many items were shown between these two?'));

      var pairWrap = document.createElement('div');
      pairWrap.style.cssText = 'display:flex;flex-direction:row;justify-content:center;align-items:center;gap:80px;margin-bottom:20px;';

      var l = createCardImg(left_img, 'medium');
      l.img.id = 'dist-left-img';
      var r = createCardImg(right_img, 'medium');
      r.img.id = 'dist-right-img';
      pairWrap.appendChild(l.card);
      pairWrap.appendChild(r.card);
      wrap.appendChild(pairWrap);

      var inputWrap = document.createElement('div');
      inputWrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;min-height:110px;';
      var input = document.createElement('input');
      input.type = 'number';
      input.min = '0';
      input.max = '9';
      input.id = 'distance-input';
      input.maxLength = 1;
      input.style.cssText = 'font-size:22px;padding:8px;width:200px;text-align:center;margin-top:18px;-moz-appearance:textfield;appearance:textfield;';
      inputWrap.appendChild(input);
      inputWrap.appendChild(createSubtext('Press 0-9 to submit your answer.'));

      var errorEl = document.createElement('p');
      errorEl.id = 'distance-error';
      errorEl.style.cssText = 'color:#d9534f;font-size:18px;margin-top:10px;display:none;';
      inputWrap.appendChild(errorEl);
      wrap.appendChild(inputWrap);

      display_element.innerHTML = '';
      display_element.appendChild(wrap);

      if (!document.getElementById('memory-distance-input-style')) {
        var style = document.createElement('style');
        style.id = 'memory-distance-input-style';
        style.innerHTML = '#distance-input::-webkit-outer-spin-button,#distance-input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}';
        document.head.appendChild(style);
      }

      var start_time = performance.now();
      var submitted = false, locked = false;

      var loaded = 0;
      function onLoad() {
        loaded++;
        if (loaded >= 2) {
          wrap.style.visibility = 'visible';
          input.focus();
        }
      }

      l.img.addEventListener('load', onLoad);
      r.img.addEventListener('load', onLoad);
      if (l.img.complete) onLoad();
      if (r.img.complete) onLoad();

      function showError(msg) {
        errorEl.textContent = msg;
        errorEl.style.display = 'block';
      }

      function handleSubmit() {
        if (submitted || locked) return;
        var v = parseFloat(input.value);
        if (isNaN(v) || input.value === '') return;
        if (v < 0) v = 0;
        if (v > 9) {
          showError('Value must be 9 or below.');
          return;
        }

        submitted = true;
        clearTimeout(distTimeoutID);

        if (window._memoryDistanceKeyHandler) {
          document.removeEventListener('keydown', window._memoryDistanceKeyHandler);
          delete window._memoryDistanceKeyHandler;
        }

        window.__memorySuppressKeysUntil = performance.now() + 350;
        distance_rt = Math.round(performance.now() - start_time);

        if (hasSlider) {
          renderSliderScreen(v, true_distance);
        } else {
          finishTrial(v, true_distance, false);
        }
      }

      input.addEventListener('keydown', function(e) {
        if (locked) { e.preventDefault(); return; }
        if (e.key === 'Backspace' || e.key === 'Delete' || e.key === 'Tab') return;
        if (e.ctrlKey || e.metaKey) return;
        if (/^[0-9]$/.test(e.key)) {
          e.preventDefault();
          input.value = e.key;
          handleSubmit();
          return;
        }
        e.preventDefault();
      });

      input.addEventListener('wheel', function(e) { e.preventDefault(); });
      input.addEventListener('paste', function(e) { e.preventDefault(); });

      window._memoryDistanceKeyHandler = function(e) {
        if (submitted || locked) return;
        if (/^[0-9]$/.test(e.key)) {
          e.preventDefault();
          input.value = e.key;
          handleSubmit();
        }
      };

      document.addEventListener('keydown', window._memoryDistanceKeyHandler);

      var distTimeoutID = setTimeout(function() {
        if (!submitted) {
          timed_out = true;
          locked = true;
          wrap.style.border = '4px solid red';
          input.disabled = true;

          var msg = document.createElement('p');
          msg.style.cssText = 'font-size:24px;margin-top:20px;color:#d9534f;';
          msg.textContent = 'No selection made. Please make a selection within the allowed time.';
          wrap.appendChild(msg);

          if (window._memoryDistanceKeyHandler) {
            document.removeEventListener('keydown', window._memoryDistanceKeyHandler);
            delete window._memoryDistanceKeyHandler;
          }

          setTimeout(function() {
            if (distance_attempt_number < 2) {
              distance_attempt_number++;
              distancePromptShown = false;
              renderDistancePrompt(true);
            } else {
              if (hasSlider) renderSliderScreen(null, true_distance);
              else finishTrial(null, true_distance, false);
            }
          }, 5000);
        }
      }, 7500);
    }

    function renderSliderScreen(dist_est_value, true_distance) {
      var wrap = createBlock3Wrapper();
      wrap.id = 'slider-container';

      wrap.appendChild(createStepLabel('Where did the middle image occur?'));

      var boundaryWrap = document.createElement('div');
      boundaryWrap.className = 'b3-boundary-card-wrap';
      boundaryWrap.style.cssText = 'margin-bottom:20px;';
      var middleCard = createCardImg(middleStim, 'small');
      boundaryWrap.appendChild(middleCard.card);
      wrap.appendChild(boundaryWrap);

      var lineShell = document.createElement('div');
      lineShell.className = 'b3-line-shell slider-shell';
      lineShell.style.cssText = 'position:relative;width:80%;margin:0 auto 20px;';

      var line = document.createElement('div');
      line.className = 'b3-line slider-line';
      line.style.cssText = 'position:relative;display:flex;align-items:center;justify-content:space-between;';

      var firstImg = (true_first_idx === idx1) ? stim1 : stim2;
      var secondImg = (true_first_idx === idx1) ? stim2 : stim1;

      var leftAnchor = createCardImg(firstImg, 'small');
      leftAnchor.card.classList.add('b3-anchor-card', 'left-anchor');
      leftAnchor.card.style.cssText += 'flex-shrink:0;';
      line.appendChild(leftAnchor.card);

      var sliderWrap = document.createElement('div');
      sliderWrap.className = 'b3-slider-wrap';
      sliderWrap.style.cssText = 'flex:1;margin:0 16px;';
      var slider = document.createElement('input');
      slider.className = 'b3-slider';
      slider.type = 'range';
      slider.min = '0';
      slider.max = '100';
      slider.step = '1';
      slider.value = '50';
      slider.style.cssText = 'width:100%;';
      sliderWrap.appendChild(slider);
      line.appendChild(sliderWrap);

      var rightAnchor = createCardImg(secondImg, 'small');
      rightAnchor.card.classList.add('b3-anchor-card', 'right-anchor');
      rightAnchor.card.style.cssText += 'flex-shrink:0;';
      line.appendChild(rightAnchor.card);

      lineShell.appendChild(line);
      wrap.appendChild(lineShell);

      var instrText = document.createElement('div');
      instrText.className = 'instruction-text padded-text prominent-subtext';
      instrText.style.cssText = 'font-size:16px;text-align:center;color:#555;margin-bottom:20px;';
      instrText.textContent = 'This image appeared sometime between these two images. Move the slider to show where you think it occurred.';
      wrap.appendChild(instrText);

      var submitBtn = document.createElement('button');
      submitBtn.style.cssText = 'font-size:18px;padding:10px 32px;cursor:pointer;border:2px solid #333;border-radius:8px;background:#fff;';
      submitBtn.textContent = 'Submit';
      wrap.appendChild(submitBtn);

      display_element.innerHTML = '';
      display_element.appendChild(wrap);

      var start_time = performance.now();

      function updateThumb() {
        var v = parseInt(slider.value, 10);
        slider.style.setProperty('--slider-value', v + '%');
      }

      slider.addEventListener('input', updateThumb);
      updateThumb();

      submitBtn.addEventListener('click', function() {
        placement_slider_value = parseInt(slider.value, 10);
        placement_rt = Math.round(performance.now() - start_time);
        finishTrial(dist_est_value, true_distance, false);
      });

      setTimeout(function() {
        if (placement_slider_value === null) {
          placement_slider_value = parseInt(slider.value, 10);
          placement_rt = Math.round(performance.now() - start_time);
          timed_out = true;
          finishTrial(dist_est_value, true_distance, false);
        }
      }, 15000);
    }

    function finishTrial(dist_est, true_distance, skipped) {
      var placement_error = null;
      if (placement_slider_value !== null && middleItemIndex !== null) {
        var true_midpoint = 50;
        placement_error = placement_slider_value - true_midpoint;
      }

      var placement_type = 'none';
      if (hasSlider) {
        if (pairInList(pair, BOUNDARY_MIDDLE_PAIRS_LOCAL)) {
          placement_type = 'boundary-middle';
        } else if (pairInList(pair, NONBOUNDARY_MIDDLE_PAIRS_LOCAL)) {
          placement_type = 'nonboundary-middle';
        } else {
          placement_type = 'slider-uncategorized';
        }
      }

      var trial_data = {
        task_phase: 'memory',
        block: block,
        true_vol_param: _trueVolParam,
        true_stc_param: _trueStcParam,
        valence: _valence,
        vol_level: _volLevel,
        stc_level: _stcLevel,
        condition: _condition,
        pair_index: trial.pair_index,
        trial1_index: idx1,
        trial2_index: idx2,
        stim_left_img: left_img,
        stim_right_img: right_img,
        stim_first_actual: true_first_img,
        order_choice_side: order_choice_side,
        order_choice_img: order_choice_img,
        order_correct: order_correct,
        order_correct_bin: (order_correct === null ? null : (order_correct ? 1 : 0)),
        order_rt: order_rt,
        distance_estimate: dist_est,
        distance_rt: distance_rt,
        pair_true_distance: true_distance,
        attempt_number: attempt_number,
        timed_out: timed_out,
        skipped_pair: skipped || false,

        placement_slider_value: placement_slider_value,
        placement_rt: placement_rt,
        middle_item_index: middleItemIndex,
        middle_item_img: middleStim,
        middle_item_is_boundary:
          (placement_type === 'boundary-middle' ? 1 :
          placement_type === 'nonboundary-middle' ? 0 : null),
        placement_trial_type: placement_type,
        placement_error_from_true_midpoint: placement_error
      };

      display_element.innerHTML = '';
      jsPsych.finishTrial(trial_data);
    }

    renderOrderScreen();
  };

  return plugin;
})();
