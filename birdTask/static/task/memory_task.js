/*
 * jsPsych plugin for a two‑part memory test used in the bird game (MOD13 V2).
 *
 */

jsPsych.plugins['memory-task'] = (function() {
  var plugin = {};

  plugin.info = {
    name: 'memory-task',
    description: 'Two‑stage temporal order and distance memory test with fixed pairs and enhanced controls.',
    parameters: {
      /**
       * Ordinal position of this memory trial within the timeline.  It is
       * stored with the data but does not influence which pair is shown.
       */
      pair_index: {
        type: jsPsych.plugins.parameterType.INT,
        default: undefined
      },
      /**
       * Block number (1–n) corresponding to the block of game trials from
       * which the paired images were drawn.  Used to filter the correct
       * set of game trials and to maintain per‑block pair sequences.
       */
      block_num: {
        type: jsPsych.plugins.parameterType.INT,
        default: undefined
      },
    }
  };

  // Fixed list of trial number pairs.  Each entry refers to 1‑based
  // indices within the preceding block of 50 game trials.  These
  // pairs must appear exactly once per block.  Do not modify
  // without also updating acceptance tests.
  var PREDEFINED_PAIRS = [
    [2, 4], [6, 9], [7, 10], [11, 13], [16, 18], [17, 20],
    [22, 24], [23, 26], [28, 31], [30, 33], [34, 36], [35, 38],
    [39, 41], [42, 45]
  ];

  // Expose the pair list so the experiment timeline can size the memory
  // block correctly without duplicating or randomising pairs here.
  if (typeof window !== 'undefined') {
    window.PREDEFINED_PAIRS = PREDEFINED_PAIRS;
  }

  // For each block number maintain a shuffled copy of PREDEFINED_PAIRS.  A
  // separate pointer tracks how many pairs have been consumed.  This
  // ensures each pair is shown exactly once per block.  When all
  // pairs have been used, subsequent calls will skip the trial.
  var pairOrderByBlock = {};
  var pairProgressByBlock = {};

  /**
   * Retrieve the next unused pair for a given block.  If the order or
   * progress have not yet been initialised, a shuffled order is
   * created and the progress pointer is reset.  If no pairs remain,
   * null is returned to signal that no further memory trials should
   * present images.
   *
   * @param {number} block The current block number
   * @returns {Array<number>|null} The next pair [idx1, idx2] or null
   */
  function nextPairForBlock(block) {
    if (!pairOrderByBlock.hasOwnProperty(block)) {
      pairOrderByBlock[block] = jsPsych.randomization.shuffle(PREDEFINED_PAIRS.slice());
      pairProgressByBlock[block] = 0;
    }
    var order = pairOrderByBlock[block];
    var progress = pairProgressByBlock[block];
    if (progress >= order.length) {
      return null;
    }
    var pair = order[progress];
    pairProgressByBlock[block] = progress + 1;
    return pair;
  }

  plugin.trial = function(display_element, trial) {
      // Clear any background images applied by other tasks.
      document.body.style.backgroundImage = '';
      document.body.style.backgroundSize = '';
      document.body.style.backgroundPosition = '';
      document.body.style.backgroundRepeat = '';

      var block = trial.block_num;
      // --- Recover block-level uncertainty parameters for this participant ---
      // We expect an earlier jsPsych data row to store arrays true_vol and true_stc (length 4),
      // corresponding to the randomized order of blocks for this participant.
      function _getBlockParamsFromData() {
        try {
          var rec = jsPsych.data.get()
            .filterCustom(function(r) { return Array.isArray(r.true_vol) && r.true_vol.length; })
            .last(1)
            .values()[0];
          if (rec && Array.isArray(rec.true_vol) && Array.isArray(rec.true_stc)) {
            return { true_vol: rec.true_vol, true_stc: rec.true_stc };
          }
        } catch (e) {}
        return { true_vol: null, true_stc: null };
      }

      var _bp = _getBlockParamsFromData();
      var _trueVolParam = (Array.isArray(_bp.true_vol) && _bp.true_vol.length >= block) ? _bp.true_vol[block - 1] : null;
      var _trueStcParam = (Array.isArray(_bp.true_stc) && _bp.true_stc.length >= block) ? _bp.true_stc[block - 1] : null;
      var _volLevel = (_trueVolParam === null) ? null : (_trueVolParam === 4 ? 'low' : 'high');
      var _stcLevel = (_trueStcParam === null) ? null : (_trueStcParam === 16 ? 'low' : 'high');
      var _condition = (_trueVolParam === null || _trueStcParam === null) ? null : ('vol' + _trueVolParam + '_stc' + _trueStcParam);

      var pair = nextPairForBlock(block);
      // If no pairs remain, end the trial immediately.  Record minimal
      // information and a skip flag to aid downstream analysis.
      if (pair === null) {
        var skip_data = {
          task_phase: 'memory',
          block: block,
          true_vol_param: _trueVolParam,
          true_stc_param: _trueStcParam,
          vol_level: _volLevel,
          stc_level: _stcLevel,
          condition: _condition,
          order_correct_bin: null,
          pair_index: trial.pair_index,
          trial1_index: null,
          trial2_index: null,
          stim_left_img: null,
          stim_right_img: null,
          stim_first_actual: null,
          order_choice_side: null,
          order_choice_img: null,
          order_correct: null,
          order_rt: null,
          distance_estimate: null,
          distance_rt: null,
          pair_true_distance: null,
          attempt_number: null,
          timed_out: null,
          skipped_pair: true
        };
        jsPsych.finishTrial(skip_data);
        return;
      }

      // Extract 1‑based indices for the two images in this pair.
      var idx1 = pair[0];
      var idx2 = pair[1];

      // Retrieve the 50 game trials for this block from the jsPsych data.
      // Filter by trial_type and block, and exclude practice trials by checking
      // for experimental metadata (true_vol_param is only set for main trials, not practice).
      var block_trials = jsPsych.data.get().filter({ trial_type: 'trial', block: block })
        .filterCustom(function(trial) {
          return trial.true_vol_param !== null && trial.true_vol_param !== undefined;
        }).values();
      // Sort trials by their trial number (1-50) to ensure correct indexing
      block_trials.sort(function(a, b) {
        return (a.trial || 0) - (b.trial || 0);
      });
      if (block_trials.length < 50) {
        console.warn('memory-task plugin: expected 50 trials for block', block,
                     'but found', block_trials.length, '. Memory responses may be misaligned.');
      }
      // Determine stimulus image paths (convert 1‑based indices to 0‑based array positions).
      // After sorting, block_trials[0] is trial 1, block_trials[1] is trial 2, etc.
      var stim1 = block_trials[idx1 - 1] ? block_trials[idx1 - 1].stim_img : undefined;
      var stim2 = block_trials[idx2 - 1] ? block_trials[idx2 - 1].stim_img : undefined;
      // Determine which image truly appeared first.
      var true_first_idx = Math.min(idx1, idx2);
      var true_first_img = (true_first_idx === idx1) ? stim1 : stim2;
      // Randomly assign images to left/right positions.  This ordering is
      // preserved across repeated attempts within the same trial.
      var order_images = jsPsych.randomization.shuffle([stim1, stim2]);
      var left_img = order_images[0];
      var right_img = order_images[1];

      // Variables to track responses and timing across attempts.
      var order_choice_side = null;
      var order_choice_img = null;
      var order_correct = null;
      var order_rt = null;
      var distance_rt = null;
      var attempt_number = 1;
      // Track whether any timeout occurred during the trial.  This flag
      // will be set by either the order or distance timeouts.
      var timed_out = false;
      var responded = false;
      var orderTimeoutID = null;
      // Maintain a separate attempt counter for the distance prompt.  This
      // mirrors the two‑attempt logic used for the order question so that
      // participants who do not provide a distance estimate within the
      // allotted time receive one warning and a second chance.
      var distance_attempt_number = 1;
      // Guard against accidental duplicate rendering of the distance prompt
      // within the same memory trial.
      var distancePromptShown = false;

      // Helper to render the order judgement screen.  This function is
      // called on each attempt and resets the response state.
      function renderOrderScreen() {
        responded = false;
        // Build HTML: question, images and instruction.  Replace
        // “stimuli” with “images”.
        var html = '';
        // Constrain the width of the container so that the red timeout
        // border does not span the full viewport.  The width is set
        // relative to the screen (80%) and centered via auto margins.
        html += '<div class="memory-container" id="order-container" ';
        html += 'style="display:flex; flex-direction:column; align-items:center; ' +
          'justify-content:center; height:80vh; width:100%; margin:0 auto; visibility:hidden;">';
        html += '<div style="display:flex; flex-direction:row; justify-content:center; align-items:center; gap:80px; margin-bottom:20px;">';
        html += '<div style="display:flex; flex-direction:column; align-items:center;">';
        html += '<img src="' + left_img + '" id="left-img" style="max-width:300px; height:auto;" />';
        html += '<div style="font-size:24px; font-weight:bold; margin-top:10px; color:#333;">1</div>';
        html += '</div>';
        html += '<div style="display:flex; flex-direction:column; align-items:center;">';
        html += '<img src="' + right_img + '" id="right-img" style="max-width:300px; height:auto;" />';
        html += '<div style="font-size:24px; font-weight:bold; margin-top:10px; color:#333;">2</div>';
        html += '</div>';
        html += '</div>';
        html += '<p style="font-size:28px; text-align:center; margin-bottom:20px; min-height:42px;">Which item came first?</p>';
        html += '<div style="display:flex; flex-direction:column; align-items:center; justify-content:flex-start; min-height:110px;">';
        html += '<p style="font-size:16px; text-align:center; color:#555; margin:0;">Press 1 or 2 to select.</p>';
        html += '</div>';
        html += '</div>';
        display_element.innerHTML = html;
        // Record start time for this attempt.
        var start_time_order = performance.now();
        var imagesLoaded = 0;
        var totalImages = 2;
        var containerEl = display_element.querySelector('#order-container');
        // Show container once both images are loaded
        function showIfReady() {
          imagesLoaded++;
          if (imagesLoaded === totalImages && containerEl) {
            containerEl.style.visibility = 'visible';
          }
        }
        // Attach load handlers and keyboard handler
        display_element.querySelector('#left-img').addEventListener('load', showIfReady);
        display_element.querySelector('#right-img').addEventListener('load', showIfReady);
        
        // Wire keyboard input (1 and 2 keys)
        function keyboardHandler(e) {
          // Ignore held-key repeats and key events that immediately follow
          // submission on the previous distance prompt.
          if (e.repeat) {
            return;
          }
          if (window.__memorySuppressKeysUntil && performance.now() < window.__memorySuppressKeysUntil) {
            e.preventDefault();
            return;
          }

          if (!responded) {
            if (e.key === '1') {
              e.preventDefault();
              if (!responded) {
                responded = true;
                clearTimeout(orderTimeoutID);
                order_rt = Math.round(performance.now() - start_time_order);
                order_choice_side = 'left';
                order_choice_img = left_img;
                order_correct = (order_choice_img === true_first_img);
                // Remove keyboard handler
                document.removeEventListener('keydown', keyboardHandler);
                // Proceed to distance prompt.
                renderDistancePrompt();
              }
            } else if (e.key === '2') {
              e.preventDefault();
              if (!responded) {
                responded = true;
                clearTimeout(orderTimeoutID);
                order_rt = Math.round(performance.now() - start_time_order);
                order_choice_side = 'right';
                order_choice_img = right_img;
                order_correct = (order_choice_img === true_first_img);
                // Remove keyboard handler
                document.removeEventListener('keydown', keyboardHandler);
                renderDistancePrompt();
              }
            }
          }
        }
        document.addEventListener('keydown', keyboardHandler);
        // Set a 5‑second timeout for this attempt.  If it fires, show
        // the same notice used in the delay discounting task and either
        // repeat or skip depending on attempt count.
        orderTimeoutID = setTimeout(function() {
          if (!responded) {
            responded = true;
            timed_out = true;
            // Remove keyboard handler on timeout
            document.removeEventListener('keydown', keyboardHandler);
            // Create a red border around the container and display the
            // timeout message identical to the delay discounting task.
            var containerEl = display_element.querySelector('.memory-container');
            if (containerEl) {
              containerEl.style.border = '4px solid red';
              // Create message element
              var msg = document.createElement('p');
              msg.style.fontSize = '24px';
              msg.style.marginTop = '20px';
              msg.style.color = '#d9534f';
              msg.textContent = 'No selection made. Please make a selection within the allowed time.';
              containerEl.appendChild(msg);
            }
            // After 3.5 seconds decide whether to repeat or skip.
            setTimeout(function() {
              // Remove the message and border for the next screen.
              if (containerEl) {
                containerEl.style.border = '';
                if (msg && msg.parentNode) {
                  msg.parentNode.removeChild(msg);
                }
              }
              if (attempt_number < 2) {
                // Increment attempt and re‑render the same order screen.
                attempt_number += 1;
                renderOrderScreen();
              } else {
                // Skip distance prompt for this pair and finish trial.
                finishTrial(null, Math.abs(idx2 - idx1) - 1, true /*skipped*/);
              }
            }, 5000);
          }
        }, 7500);
      }

      // Helper to render the distance estimation prompt.  Shows the two
      // images along with a numeric input and handles validation and
      // submission.  Participants must enter a value between 0 and 9.
      function renderDistancePrompt(forceRerender) {
        if (distancePromptShown && forceRerender !== true) {
          return;
        }
        distancePromptShown = true;
        // Compute true distance between the two selected trials.
        var true_distance = Math.abs(idx2 - idx1) - 1;
        // Build HTML: keep images visible; add instructions and input.
        var html = '';
        // Constrain the width of the container similar to the order screen and
        // center it so that the timeout border appears around the content
        // rather than spanning the full page.
        html += '<div class="memory-distance" id="distance-container" ' +
          'style="display:flex; flex-direction:column; align-items:center; justify-content:center; ' +
          'height:80vh; width:100%; margin:0 auto; visibility:hidden;">';
        html += '<div style="display:flex; flex-direction:row; justify-content:center; ' +
          'align-items:center; gap:80px; margin-bottom:20px;">';
        html += '<div style="display:flex; flex-direction:column; align-items:center;">';
        html += '<img src="' + left_img + '" id="dist-left-img" style="max-width:300px; height:auto;" />';
        html += '<div style="font-size:24px; font-weight:bold; margin-top:10px; color:transparent;">1</div>';
        html += '</div>';
        html += '<div style="display:flex; flex-direction:column; align-items:center;">';
        html += '<img src="' + right_img + '" id="dist-right-img" style="max-width:300px; height:auto;" />';
        html += '<div style="font-size:24px; font-weight:bold; margin-top:10px; color:transparent;">2</div>';
        html += '</div>';
        html += '</div>';
        html += '<p style="font-size:28px; text-align:center; margin-bottom:20px; min-height:42px;">How many items were shown between these two?</p>';
        html += '<div style="display:flex; flex-direction:column; align-items:center; justify-content:flex-start; min-height:110px;">';
        html += '<input type="number" min="0" max="9" id="distance-input" maxlength="1" ' +
          'autofocus style="font-size:22px; padding:8px; width:200px; text-align:center; margin-top:18px;" />';
        html += '<p style="font-size:16px; text-align:center; color:#555; margin-top:20px; margin-bottom:0;">Press 0-9 to submit your answer.</p>';
        html += '</div>';
        html += '<p id="distance-error" style="color:#d9534f; font-size:18px; margin-top:10px; display:none;"></p>';
        html += '</div>';
        display_element.innerHTML = html;
        
        // Add CSS to hide number input spinner
        if (!document.getElementById('memory-distance-input-style')) {
          var style = document.createElement('style');
          style.id = 'memory-distance-input-style';
          style.innerHTML =
            '#distance-input::-webkit-outer-spin-button, #distance-input::-webkit-inner-spin-button { ' +
            '-webkit-appearance: none; margin: 0; } ' +
            '#distance-input { -moz-appearance: textfield; appearance: textfield; }';
          document.head.appendChild(style);
        }
        
        var start_time_dist = performance.now();
        var submitted = false;
        var distanceLocked = false;
        
        // Clean up any lingering keyboard handlers from previous tasks (e.g., discounting)
        if (window._ddKeyboardHandler) {
          document.removeEventListener('keydown', window._ddKeyboardHandler);
          delete window._ddKeyboardHandler;
        }
        
        // Show container once both images are loaded
        var distImagesLoaded = 0;
        var distTotalImages = 2;
        var distContainerEl = display_element.querySelector('#distance-container');
        function showDistIfReady() {
          distImagesLoaded++;
          if (distImagesLoaded === distTotalImages && distContainerEl) {
            distContainerEl.style.visibility = 'visible';
            // Focus the input field after the container becomes visible
            if (distanceInput) {
              distanceInput.focus();
            }
          }
        }
        display_element.querySelector('#dist-left-img').addEventListener('load', showDistIfReady);
        display_element.querySelector('#dist-right-img').addEventListener('load', showDistIfReady);
        // Will hold the timeout ID for the distance prompt so it can be
        // cancelled if the participant submits before the timeout fires.
        var distTimeoutID;
        var distanceInput = display_element.querySelector('#distance-input');
        var errorEl = display_element.querySelector('#distance-error');
        // Ensure no stale distance key handler is left from a prior attempt.
        if (window._memoryDistanceKeyHandler) {
          document.removeEventListener('keydown', window._memoryDistanceKeyHandler);
          delete window._memoryDistanceKeyHandler;
        }
        // Helper to show an error message and keep prompt on screen.
        function showError(msg) {
          errorEl.textContent = msg;
          errorEl.style.display = 'block';
        }
        function hideError() {
          errorEl.style.display = 'none';
        }
        // Submission handler.  Validates input then ends trial.
        function handleSubmit() {
          if (submitted || distanceLocked) return;
          var valStr = distanceInput.value;
          // If empty, do not accept.  Show message and return.
          if (valStr === '' || valStr === null) {
            showError('Please enter a number.');
            return;
          }
          var valNum = parseFloat(valStr);
          if (isNaN(valNum)) {
            showError('Please enter a valid number.');
            return;
          }
          if (valNum < 0) {
            valNum = 0;
          }
          if (valNum > 9) {
            showError('Value must be 9 or below. Please try again');
            return;
          }
          // All validations passed; hide any error and finish.  Cancel any
          // pending timeout so the warning does not fire while processing.
          hideError();
          submitted = true;
          if (window._memoryDistanceKeyHandler) {
            document.removeEventListener('keydown', window._memoryDistanceKeyHandler);
            delete window._memoryDistanceKeyHandler;
          }
          if (typeof distTimeoutID !== 'undefined') {
            clearTimeout(distTimeoutID);
          }
          // Prevent this same keypress (especially 1/2) from being consumed
          // by the next order screen when the next trial renders.
          window.__memorySuppressKeysUntil = performance.now() + 350;
          distance_rt = Math.round(performance.now() - start_time_dist);
          finishTrial(valNum, true_distance, false /*skipped*/);
        }
        // Keyboard input handler: only allow digit keys (0-9), Backspace,
        // Delete and Tab. Digit keypress submits immediately. Disallow arrow keys and mouse
        // actions (wheel/paste) so the on-screen arrows cannot be used.
        distanceInput.addEventListener('keydown', function(e) {
          if (distanceLocked) {
            e.preventDefault();
            return;
          }
          // Allow common editing/navigation keys
          if (e.key === 'Backspace' || e.key === 'Delete' || e.key === 'Tab') {
            return;
          }
          // Allow modifier combos (Ctrl/Cmd) so shortcuts still work.
          if (e.ctrlKey || e.metaKey) {
            return;
          }
          // Only allow single-digit keys
          if (/^[0-9]$/.test(e.key)) {
            e.preventDefault();
            distanceInput.value = e.key;
            handleSubmit();
            return;
          }
          // Otherwise block the key
          e.preventDefault();
        });
        // Prevent mouse wheel from changing value
        distanceInput.addEventListener('wheel', function(e) { e.preventDefault(); });
        // Prevent paste/drop so only keyboard digit entry is possible
        distanceInput.addEventListener('paste', function(e) { e.preventDefault(); });
        distanceInput.addEventListener('drop', function(e) { e.preventDefault(); });
        // Enforce single digit: trim value to one character on input
        distanceInput.addEventListener('input', function(e) {
          if (distanceInput.value.length > 1) {
            distanceInput.value = distanceInput.value.slice(0, 1);
          }
        });
        // Fallback keyboard handler so digit keys work even if the input
        // loses focus (e.g., after a keypress or image load timing).
        window._memoryDistanceKeyHandler = function(e) {
          if (submitted || distanceLocked) return;
          if (e.key === 'Backspace' || e.key === 'Delete') {
            e.preventDefault();
            distanceInput.value = '';
            return;
          }
          if (/^[0-9]$/.test(e.key)) {
            e.preventDefault();
            distanceInput.value = e.key;
            handleSubmit();
            return;
          }
        };
        document.addEventListener('keydown', window._memoryDistanceKeyHandler);
        // Focus the input for convenience.
        distanceInput.focus();

        // Apply a 5‑second timeout for the distance question.  If the
        // participant does not submit an answer within this period,
        // display the same red border and warning message used in the
        // order judgement and either repeat the prompt or finish the
        // trial after a second timeout.
        var distContainerEl = display_element.querySelector('.memory-distance');
        var distMsg;
        distTimeoutID = setTimeout(function() {
          if (!submitted) {
            timed_out = true;
            distanceLocked = true;
            // Show border and message
            if (distContainerEl) {
              distContainerEl.style.border = '4px solid red';
              // Disable clicks on the images while showing the message
              var distLeftImg = display_element.querySelector('#dist-left-img');
              var distRightImg = display_element.querySelector('#dist-right-img');
              if (distLeftImg) distLeftImg.style.pointerEvents = 'none';
              if (distRightImg) distRightImg.style.pointerEvents = 'none';
              // Disable the input
              distanceInput.disabled = true;
              distMsg = document.createElement('p');
              distMsg.style.fontSize = '24px';
              distMsg.style.marginTop = '20px';
              distMsg.style.color = '#d9534f';
              distMsg.textContent = 'No selection made. Please make a selection within the allowed time.';
              distContainerEl.appendChild(distMsg);
            }
            if (window._memoryDistanceKeyHandler) {
              document.removeEventListener('keydown', window._memoryDistanceKeyHandler);
            }
            // Remove the message after 3.5 seconds and decide whether to
            // repeat or finish the trial based on the attempt count.
            setTimeout(function() {
              if (distContainerEl) {
                distContainerEl.style.border = '';
                if (distMsg && distMsg.parentNode) {
                  distMsg.parentNode.removeChild(distMsg);
                }
              }
              if (window._memoryDistanceKeyHandler) {
                document.removeEventListener('keydown', window._memoryDistanceKeyHandler);
                delete window._memoryDistanceKeyHandler;
              }
              if (distance_attempt_number < 2) {
                distance_attempt_number += 1;
                // Re‑render the distance prompt for the second attempt.
                renderDistancePrompt(true);
              } else {
                // Two timeouts: finish the trial with a null estimate.
                finishTrial(null, true_distance, false /*skipped*/);
              }
            }, 5000);
          }
        }, 7500);
      }

      // Finish the trial and store data.  Accepts the participant’s
      // numeric estimate (or null), the true distance between the two
      // images, and a flag indicating whether the pair was skipped due
      // to two timeouts.
      function finishTrial(dist_est, true_distance, skipped) {
        // Construct the data record for this memory trial.
        var trial_data = {
          task_phase: 'memory',
          block: block,
          true_vol_param: _trueVolParam,
          true_stc_param: _trueStcParam,
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
          skipped_pair: skipped || false
        };
        display_element.innerHTML = '';
        jsPsych.finishTrial(trial_data);
      }
      // Helper to capture the current attempt number when finishing.  If
      // the participant times out twice, the attempt_number recorded
      // should reflect the attempt that led to the final outcome.
      function attemptedTime() {
        return attempt_number;
      }
      // Start the first order judgement attempt.
      renderOrderScreen();
  };
  return plugin;
})();