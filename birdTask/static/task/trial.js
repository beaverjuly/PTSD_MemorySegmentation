jsPsych.plugins["trial"] = (function () {

  var plugin = {};

  plugin.info = {
    name: 'trial',
    description: '',
    parameters: {
      dummy: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'dummy',
        default: null,
        description: 'starting location'
      },
      terminate_now: {
        type: jsPsych.plugins.parameterType.BooL,
        array: false,
        default: false,
      },
      show_missing: {
        type: jsPsych.plugins.parameterType.BooL,
        array: false,
        default: false,
      },
      show_bird: {
        type: jsPsych.plugins.parameterType.BooL,
        array: false,
        default: false,
      },
      is_moving_practice: {
        type: jsPsych.plugins.parameterType.BooL,
        array: false,
        default: false,
      },
      canvas_size: {
        type: jsPsych.plugins.parameterType.INT,
        array: true,
        pretty_name: 'Canvas size',
        default: [2000, 2000],
        description: 'Array containing the height (first value) and width (second value) of the canvas element.'
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: 'Choices',
        default: [32, 37,39],
        description: 'Keys corresponding to each context (left, right, down).'
      },
      strong_warning: {
        type: jsPsych.plugins.parameterType.BooL,
        array: false,
        default: false,
      },
      missing_msg_warning_number: {
        type: jsPsych.plugins.parameterType.BooL,
        array: false,
        default: 15,
      },
      bucket_position: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: null,
        description: 'location of the bucket.'
      },
      stayed: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 1,
        description: '0 is the bucket is mvoed in this trial, otherwise 1.'
      },
      coins_distribution: {
        type: jsPsych.plugins.parameterType.INT,
        array: true,
        pretty_name: 'distribution of coins',
        default: [-10, -5, 0, 5, 10],
        description: 'distribution of coins.'
      },
      coins_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: true,
        default: [600, 400, 0, 400, 600],
        description: 'falling duration of coins.'
      },
      bag_position: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        pretty_name: 'x of coin bag',
        default: 35,
        description: 'x of coin bag'
      },
      bird_position: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: null,
        description: 'x of the bird'
      },
      no_response_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 1500,
      },
      response_remaining_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 2000,
      },
      drop_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        // Shorten the default trial duration when the bag drop is accelerated.
        // With a 50 % faster bag drop the overall drop sequence completes
        // earlier, so reduce the drop duration accordingly to maintain
        // alignment between the coin animations and the end of the trial.
        default: 5000,
      },
      missing_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 7500,
      },
      /**
       * Stimulus image to display after the coins drop. This should be the path
       * to a file containing a static image. It will appear at the location
       * where the bag drops and remain on screen for a short time before
       * disappearing.
       */
      stim_img: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null,
        description: 'Path to the stimulus image for the memory component.'
      },
      background_img: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null,
        description: 'Path to the background image used to cue the block context.'
      },
      land_img: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null,
        description: 'Path to the land background image for this block.'
      }
      ,
      /**
       * Whether to hide the stimulus image during the memory introduction.
       * When true the stimulus will still be present in the DOM (so that
       * measurements such as width can be computed for alignment) but it
       * will not be visible to participants. This allows early practice
       * trials to include the image element for proper coin counter
       * placement without actually displaying the item.
       */
      hide_stimulus: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false,
        description: 'If true, keep the stimulus hidden even after the coins drop.'
      }
    }
  }

  var make_html = function(trial) {
    // The jsPsych canvas element is not used in this custom trial to avoid covering gameplay
    var new_html = '';

    trial.dummy = trial.coins_distribution.length;

    var containerStyleAttr = '';
    if (trial.background_img) {
      containerStyleAttr = `style="background-image: url('${trial.background_img}'); ` +
        `background-size: cover; background-position: center; background-repeat: no-repeat;"`;  
    }

    new_html += `<style>
    body {
      height: 100vh;
      max-height: 100vh;
      overflow: hidden;
      position: fixed;
      background-color: #ffffff;
    }
    </style>`;

    new_html += `<div class="game-container">`;

      
    // LAYERED ATTEMPT
      new_html += `<div class="bg-sky"></div>`;

    // Lock overlay (always dims sky, sits above sky but under land)
      new_html += `<div class="light1" id="light"></div>`;

    // Dynamic land background (switch per block)
      if (trial.land_img) {
        new_html += `<div class="bg-land" style="
          background-image: url('${trial.land_img}');
          background-size: cover;
          background-position: center bottom;
          background-repeat: no-repeat;">
        </div>`;
      } else {
        new_html += `<div class="bg-land"></div>`;
      }

    // Open the main container and attach the per‑trial backdrop (if any).

    new_html += `<div class="main-container">`;
    //new_html += '<div class="light1" id="light"></div>';
      new_html += `<div class="bucket1" id="bucket" style="left: ${trial.bucket_position}%;"></div>`;
      new_html += `<div class="bucket2" id="bucket2" style="left: ${trial.bucket_position}%; opacity: 0%;"></div>`;

      new_html += `<div class="expl1" id="expl" style="animation: none; opacity: 0%;"></div>`;
      new_html += `<div class="bag" id="bag" style="opacity: 0%; animation: none; top: 115%;"></div>`;
      new_html += `<div class="bird" number="0" id="birdy" style="visibility: hidden;"></div>`;
      new_html += `<img src="${trial.stim_img}" class="stimulus-img" id="stimulus-img" style="visibility:hidden; position:absolute;" />`;

    // Create coin elements. They will be made visible when the drop sequence starts.
    for (var i = 0; i < trial.coins_distribution.length; i++) {
      new_html += `<div class="gold" id="gold${i}" style="visibility:hidden; animation: none; top: 115%;"></div>`;
    }
    new_html += '</div>';
    new_html += '</div>';

    return new_html;
  };


  var move_bucket = function(trial, info) {
    let key = info.key;

    var bucket = document.getElementById("bucket");
    var x = trial.bucket_position;
    var step = 2;

    if (key == 39) {
      x = x + step;
    } else if (key == 37) {
      x = x- step;
    };

    var y = x;
    if (x > 90) {
      y = 90;
    } else if (x < 10) {
      y = 10;
    }

    bucket.style.left = y + "%";
    var bucket2 = document.getElementById("bucket2");
    bucket2.style.left = y + "%";

    trial.bucket_position = y;

    return trial;
  }; // end of move_bucket

  var gold_drop = function(trial) {

    var coins_duration = trial.coins_duration;

    var half_coins = (trial.coins_distribution.length-1)/2;
    for (var i = 0; i < trial.coins_distribution.length; i++) {
      var coin = document.getElementById("gold"+ (i));
      // Make the coin visible now that it is about to drop
      coin.style.visibility = "visible";
      coin.style.left = trial.bag_position + trial.coins_distribution[i] + "%";

      if ( i<half_coins) {
        coin.style.animation="drop_left";
      } else if ( i> half_coins) {
        coin.style.animation="drop_right";
      } else if (i == half_coins) {
        coin.style.animation="drop";
      }

      var dur = coins_duration[i];
      coin.style.setProperty("animation-duration", dur + "ms");
      coin.style.setProperty("animation-timing-function", "cubic-bezier(.7,.3,1,1)");
    };
    var bucket2 = document.getElementById("bucket2");
    bucket2.style.opacity = "100%";
  }; // end of gold_drop

  var fly = function(trial) {
  // first drop the bag, then explode, then coins

  let initial_duration = 300;
  // Accelerate the bag drop: halve the duration so the bag falls
  // approximately 50 % faster than in the original implementation.
  let bag_duration = 1000;
  let explode_animation = 300;
  let gold_initialization = 200;
  let explode_duration = bag_duration + gold_initialization;  
  // Declare captureIndices at function level so it's accessible to
  // both the capture check timeouts and the feedback box creation
  var captureIndices = [];  
  var light = document.getElementById("light");
  if (light) {
    light.style.opacity = "30%";
  }

  // The old layered background included a moon element to dim the scene
  // during the drop phase. That element has been removed from the
  // template, so only adjust the light overlay. If a moon element is
  // accidentally present it will be ignored. By not referencing a
  // non‑existent element we avoid console errors.
  var moon = document.getElementById("moon");
  if (moon) {
    moon.style.opacity = "30%";
  }

 setTimeout(function () {
  if (trial.show_bird) {
    var bird = document.getElementById("birdy");
    bird.style.left = trial.bird_position + "%";
    bird.style.visibility = "visible";
  }

  var bag = document.getElementById("bag");
  bag.style.left = trial.bag_position + "%";
  bag.style.animation = "bagdrop";
  bag.style.setProperty("animation-duration", bag_duration + "ms");
  bag.style.setProperty("animation-timing-function", "ease-in");

  // Do not reveal the stimulus image during the bag drop. It will be shown
  // later (after the coins have landed) by an explicit timeout. Ensure
  // that it remains hidden during the drop animation.
  var stim = document.getElementById("stimulus-img");
  if (stim) {
    stim.style.visibility = "hidden";
    stim.style.left = trial.bag_position + "%";
    stim.style.top = "80%";
  }

}, initial_duration);


  setTimeout(function() {
    bag.style.visible = "hidden";
    var expl = document.getElementById("expl");
    expl.style.left = trial.bag_position + "%";
    expl.style.animation="explode";
    expl.style.setProperty("animation-duration", explode_animation + "ms");
    // Initiate the coin drop sequence.
    gold_drop(trial);

    /*
     * Schedule disappearance for individual coins.  Coins that land in
     * the bucket should disappear before they reach the bottom of
     * the screen, while coins that miss should continue their
     * animation and disappear at the end of their drop.  We compute
     * which coins will be captured based on the horizontal distance
     * between the coin and the bucket.  A capture width of 6% is
     * retained from the feedback calculation.  For captured coins we
     * hide them partway through their animation (roughly when the
     * coin reaches the vertical position of the bucket); for all
     * coins we hide them completely when their animation has
     * finished.
     */
    // Determine capture using a pixel-overlap check scheduled at the
    // moment the coin reaches the top of the bucket (approximately
    // captureFraction through its animation). If overlap >= 50% of the
    // coin width, the coin is marked captured and hidden at that time.
    // Note: captureIndices is declared at the function level
    for (var ci2 = 0; ci2 < trial.coins_distribution.length; ci2++) {
      (function(idx) {
        var coinEl = document.getElementById('gold' + idx);
        var dur = trial.coins_duration[idx];
        // Fraction of the duration after which the coin reaches the
        // vertical position of the bucket. This value approximates the
        // timing used previously and can be tuned if needed.
        var captureFraction = 19 / 30;

        // Schedule a check at the moment the coin reaches the bucket's
        // vertical location. Use a timeout so the animation has moved
        // the coin into place and getBoundingClientRect returns the
        // correct current coordinates.
        setTimeout(function() {
          if (!coinEl) return;
          var bucketEl = document.getElementById('bucket');
          if (!bucketEl) return;
          var coinRect = coinEl.getBoundingClientRect();
          var bucketRect = bucketEl.getBoundingClientRect();
          var coinWidth = coinRect.width || 10;
          var overlap = Math.max(0, Math.min(coinRect.right, bucketRect.right) - Math.max(coinRect.left, bucketRect.left));
          if (overlap >= coinWidth) {
            // Mark as captured and hide the coin immediately (counts as caught)
            // Requires 100% of the coin to be within the bucket bounds
            captureIndices.push(idx);
            coinEl.style.visibility = 'hidden';
          }
        }, dur * captureFraction);

        // Always hide the coin when its animation completes
        setTimeout(function() {
          if (coinEl) {
            coinEl.style.visibility = 'hidden';
          }
        }, dur + 50);
      })(ci2);
    }
  }, initial_duration + explode_duration);

    // After the bag drop and explosion have occurred, and the coins have
    // started falling, wait until the longest coin animation has
    // finished before showing the memory stimulus and feedback. This
    // ensures both appear only after the coins have disappeared from
    // view.  We compute the maximum duration from trial.coins_duration
    // and add that to the initial and explosion timing.
    var maxCoinDuration = 0;
    if (trial.coins_duration && trial.coins_duration.length > 0) {
      for (var ii = 0; ii < trial.coins_duration.length; ii++) {
        if (trial.coins_duration[ii] > maxCoinDuration) {
          maxCoinDuration = trial.coins_duration[ii];
        }
      }
    }

    setTimeout(function() {
      var stimEl = document.getElementById('stimulus-img');
      if (stimEl && trial.stim_img) {
        // Position the stimulus relative to the bag drop.  Always
        // assign explicit dimensions so that even hidden stimuli
        // occupy space for feedback sizing.  Do not reveal the
        // stimulus if hide_stimulus is true; leave it hidden so it
        // preserves layout but remains invisible to participants.
        stimEl.style.left = trial.bag_position + '%';
        // Top is finalized after feedback box layout so the bottom of
        // the stimulus can be kept slightly above the feedback box.
        stimEl.style.top = '50%';
        stimEl.style.transform = 'translateX(-50%)';
        stimEl.style.width = '23%';
        stimEl.style.height = 'auto';
        if (!trial.hide_stimulus) {
          stimEl.style.visibility = 'visible';
        } else {
          // Keep the image hidden but ensure it occupies space for
          // alignment.  We use visibility:hidden instead of
          // display:none so that width calculations still work.
          stimEl.style.visibility = 'hidden';
        }
      }
      // Compute how many coins were captured using the pixel-overlap
      // checks performed earlier when coins reached the bucket.
      var captureCount = captureIndices.length;
      // Create a feedback box and append it to the main container.
      // Position it below the landing area and center it
      // horizontally relative to where the bag landed (not the
      // bucket). This way the feedback follows the falling bag.
      var fb = document.createElement('div');
      fb.id = 'feedback-box';
      fb.style.position = 'absolute';
      // Center the feedback box horizontally on the bag drop location
      fb.style.left = trial.bag_position + '%';
      // Place the box below the landing area and center horizontally.
      // Top is set in the layout pass below.
      fb.style.top = '90%';
      fb.style.transform = 'translate(-50%, -100%)';
      fb.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
      fb.style.padding = 'clamp(1px, 0.18vw, 4px) clamp(5px, 0.45vw, 10px)';
      fb.style.borderRadius = 'clamp(2px, 0.2vw, 6px)';
      fb.style.fontSize = 'clamp(16px, 1.8vw, 34px)';
      fb.style.lineHeight = '1';
      fb.style.fontWeight = 'bold';
      fb.style.width = '23%';
      fb.style.textAlign = 'center';
      fb.style.boxSizing = 'border-box';
      fb.style.zIndex = '100';
      // Colour the feedback text based on the number of coins
      // captured.  Ten coins (perfect catch) is bright neon green;
      // zero coins (complete miss) is bright red; other values use
      // the default text colour.  Neon green (#39FF14) provides high
      // contrast and conveys success, while red (#ff2b2b) draws
      // attention to a complete miss.
      if (captureCount === 10) {
        fb.style.color = '#39FF14';
      } else if (captureCount === 0) {
        fb.style.color = '#ff2b2b';
      }
      fb.innerHTML = captureCount;
      trial.coins_caught = captureCount;
      var container = document.querySelector('.main-container');
      if (container) {
        container.appendChild(fb);

      // Keep feedback and stimulus proportional and vertically coupled:
      // feedback width tracks stimulus width, and stimulus bottom sits a
      // little above the feedback top.
      if (stimEl) {
        var alignStimulusAndFeedback = function() {
          requestAnimationFrame(function () {
            var containerRect = container.getBoundingClientRect();
            var stimRect = stimEl.getBoundingClientRect();
            if (!containerRect.height || !stimRect.height) return;

            var bucketEl = document.getElementById('bucket');
            var gapPx = Math.max(containerRect.height * 0.012, 5);
            var feedbackBottomPx;
            if (bucketEl) {
              var bucketRect = bucketEl.getBoundingClientRect();
              var bucketTopPx = bucketRect.top - containerRect.top;
              feedbackBottomPx = bucketTopPx - gapPx;
            } else {
              feedbackBottomPx = containerRect.height * 0.90;
            }

            feedbackBottomPx = Math.max(20, Math.min(containerRect.height - 4, feedbackBottomPx));
            var feedbackBottomPct = (feedbackBottomPx / containerRect.height) * 100;
            fb.style.top = feedbackBottomPct + '%';

            var fbRect = fb.getBoundingClientRect();
            var feedbackTopPx = fbRect.top - containerRect.top;
            var stimTopPx = feedbackTopPx - gapPx - stimRect.height;
            var stimTopPct = (stimTopPx / containerRect.height) * 100;
            stimTopPct = Math.max(5, Math.min(80, stimTopPct));
            stimEl.style.top = stimTopPct + '%';
          });
        };

        if (stimEl.complete) {
          alignStimulusAndFeedback();
        } else {
          stimEl.addEventListener('load', alignStimulusAndFeedback, { once: true });
        }
      }

      }
      // Hide the stimulus and remove the feedback after 2.5 seconds 
      setTimeout(function() {
        if (stimEl) {
          stimEl.style.visibility = 'hidden';
        }
        if (fb.parentNode) {
          fb.parentNode.removeChild(fb);
        }
      }, 2500);

    }, initial_duration + explode_duration + maxCoinDuration);
}; // end of fly

  plugin.trial = function (display_element, trial) {

    // ------------------------------------------------------------
    // Data/QC additions (per-trial condition tags + movement timing)
    // ------------------------------------------------------------
    // Define a consistent "trial onset" as the moment this plugin.trial
    // function begins execution.
    const _trial_onset_perf = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const _trial_onset_elapsed = (typeof jsPsych !== 'undefined' && jsPsych.totalTime) ? jsPsych.totalTime() : null;

    // Movement tracking (bucket movement RTs and summary stats)
    const _bucket_start_pos = trial.bucket_position;
    let _bucket_end_pos = trial.bucket_position;
    let _rt_first_move_ms = null; // ms from onset to first bucket move
    let _rt_last_move_ms  = null; // ms from onset to last bucket move
    let _num_moves = 0;

    // Helper: record a move if bucket position actually changed
    const _record_move_if_changed = function(prev_pos) {
      if (trial.bucket_position !== prev_pos) {
        const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        const ms_since_onset = now - _trial_onset_perf;
        if (_rt_first_move_ms === null) _rt_first_move_ms = ms_since_onset;
        _rt_last_move_ms = ms_since_onset;
        _num_moves += 1;
        _bucket_end_pos = trial.bucket_position;
      }
    };

    if (trial.terminate_now) {
      setTimeout(function() {
        end_trial(2);
      }, 1);
    };

    if (trial.is_moving_practice) {
      trial.no_response_duration = 5000;
    };

    new_html = make_html(trial);
    display_element.innerHTML = new_html;


    var bucket = document.getElementById("bucket");
    var bucket2 = document.getElementById("bucket2");
    bucket.style.left = trial.bucket_position + "%";
    bucket2.style.left = trial.bucket_position + "%";

    var after_response = function(info) {
      var prev_pos = trial.bucket_position;
      trial = move_bucket(trial,info);

      // Update movement summary stats
      _record_move_if_changed(prev_pos);
    };

    var after_1st_response = function(info) {
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener1);

      trial.stayed = 0;

      var prev_pos = trial.bucket_position;
      trial = move_bucket(trial,info);

      // Record first movement RT and update last-move time
      _record_move_if_changed(prev_pos);

      // setTimeout(function() {
      // }, 10);
      keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: true,
        allow_held_key: true
      });

      setTimeout(function() {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        if (!trial.is_moving_practice) {
          fly(trial);
        }
        jsPsych.pluginAPI.setTimeout(function() {
          end_trial(1);
        }, trial.drop_duration);
      }, trial.response_remaining_duration);

    }; // end of after_1st_response


    // the listener for the very 1st response (persist is false here and callback is after_1st_response)
    setTimeout(function() {
      jsPsych.pluginAPI.cancelAllKeyboardResponses;
      keyboardListener1 = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_1st_response,
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: false,
        allow_held_key: true,
      });
    }, 20);

    var missed_response = function() {

      // Kill all setTimeout handlers.
      // jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      // Display warning message.

      var msg = '<p style="font-size: 20px; line-height: 1.5em">Are you there? You have not moved the bucket for a long time.' +
        '<br><br>Please pay more attention and play with your bucket, otherwise we may end the exepriment early and reject your work.';
      if (trial.strong_warning) {
        msg = '<p style="font-size: 20px; line-height: 1.5em">Are you there? You have not moved the bucket for a long time.' +
          '<br><br>We have warned you more than ' + trial.missing_msg_warning_number + ' times. <br><br> <b>Warning: we are about to reject your work!</b>';
      }
      if (trial.is_moving_practice) {
        msg = '<p style="font-size: 20px; line-height: 1.5em">Are you there? You should try to move the bucket using left and right arrow keys.' +
          '<br><br>Please pay more attention and move the bucket, otherwise we will end the game here!';
      };

      display_element.innerHTML = msg;
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial(0);
      }, trial.missing_duration);

    }; // end of missed_response

    // function to end trial when it is time
    var end_trial = function(completed) {

      // Kill all setTimeout handlers.
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      // Reset light overlay for next trial
      var light = document.getElementById("light");
      if (light) {
        light.style.opacity = "0%";
      }

      // gather the data to store for the trial
      // Normalize coins_caught on incomplete trials (avoid undefined in exports)
      var coins_caught_value = (completed === 1) ? trial.coins_caught : null;

      // Trial duration relative to onset
      var _trial_duration_ms = null;
      try {
        var now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        _trial_duration_ms = now - _trial_onset_perf;
      } catch (e) {
        _trial_duration_ms = null;
      }

      // Movement duration (time actively moving bucket at least once)
      var movement_duration_ms = (_rt_first_move_ms !== null && _rt_last_move_ms !== null)
        ? (_rt_last_move_ms - _rt_first_move_ms)
        : null;

      // Optional miss reason for QC
      var miss_reason = null;
      if (completed === 0) miss_reason = 'no_move_timeout';
      if (completed === 2) miss_reason = 'terminated';

      var trial_data = {
        "bird_position": trial.bird_position,
        "bag_position": trial.bag_position,
        "bucket_position": trial.bucket_position,
        "completed": completed,
        "stayed": trial.stayed,
        "coins_caught": coins_caught_value,

        // --- Added for analysis/QC ---
        "bucket_start_pos": _bucket_start_pos,
        "bucket_end_pos": _bucket_end_pos,
        "num_moves": _num_moves,
        "rt_first_move_ms": _rt_first_move_ms,
        "rt_last_move_ms": _rt_last_move_ms,
        "movement_duration_ms": movement_duration_ms,
        "trial_onset_elapsed": _trial_onset_elapsed,
        "trial_duration_ms": _trial_duration_ms,
        "miss_reason": miss_reason,

        // --- Condition tags (pass these in from your timeline) ---
        "true_vol_param": (typeof trial.true_vol_param !== 'undefined') ? trial.true_vol_param : null,
        "true_stc_param": (typeof trial.true_stc_param !== 'undefined') ? trial.true_stc_param : null,
        "vol_level": (typeof trial.vol_level !== 'undefined') ? trial.vol_level : null,
        "stc_level": (typeof trial.stc_level !== 'undefined') ? trial.stc_level : null,
      };

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);

    }; // end of end_trial

    jsPsych.pluginAPI.setTimeout(function() {
      if (trial.show_missing) {
        missed_response();
      } else {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        if (!trial.is_moving_practice) {
          fly(trial);
        }
        jsPsych.pluginAPI.setTimeout(function() {
          end_trial(1);
        }, trial.drop_duration);
      };
    }, trial.no_response_duration);



  }; // end of plugin.trial

  return plugin;
})();