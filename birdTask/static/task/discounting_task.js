/*
 * delay discounting task for the bird game
 *
 * This module defines a custom jsPsych plugin and a helper function to
 * generate a series of intertemporal choice trials. Each trial presents
 * a Small‑Sooner (SS) reward available immediately and a Large‑Later
 * (LL) reward available after some delay. Participants choose the
 * option they prefer by pressing 1 or 2. After a choice, the selected option is highlighted for
 * 750 ms, followed by a 500 ms blank inter‑trial interval. Data from
 * each trial are appended to the main jsPsych dataset.
 *
 * The exported function `runDelayDiscounting(conditionTag)` builds a
 * timeline of 12 discounting trials plus 2 attention checks (plus a one‑time instruction page
 * shown only on the first call) using seeded randomization based on
 * participant identifier and call order. It returns a Promise that
 * resolves to an array of timeline nodes that can be concatenated
 * into the main experiment timeline. 
 */

(function() {
  var delayBlockCounter = 0;

  // ---------------------------------------------------------------------------
  // Utilities: seeded RNG
  // ---------------------------------------------------------------------------
  function stringToSeed(id) {
    if (typeof id === 'number') return id >>> 0;
    var str = String(id);
    var hash = 2166136261;
    for (var i = 0; i < str.length; i++) {
      hash ^= str.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function() {
      a |= 0;
      a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function generateDelays(rand, count) {
    var delays = [];
    var min = 1;
    var max = 1000;
    var minLog = Math.log10(min);
    var maxLog = Math.log10(max);
    for (var i = 0; i < count; i++) {
      var ratio = (count === 1) ? 0 : i / (count - 1);
      var baseDelay = Math.pow(10, minLog + ratio * (maxLog - minLog));
      var jitterFactor = 1 + (rand() * 0.2 - 0.1);
      var jittered = baseDelay * jitterFactor;
      var rounded = Math.round(jittered);
      if (rounded < min) rounded = min;
      if (rounded > max) rounded = max;
      delays.push(rounded);
    }
    for (var j = delays.length - 1; j > 0; j--) {
      var k = Math.floor(rand() * (j + 1));
      var tmp = delays[j];
      delays[j] = delays[k];
      delays[k] = tmp;
    }
    return delays;
  }

  function generateLLs(rand, count) {
    var LLs = [];
    var min = 5;
    var max = 100;
    var minLog = Math.log10(min);
    var maxLog = Math.log10(max);
    for (var i = 0; i < count; i++) {
      var ratio = (count === 1) ? 0 : i / (count - 1);
      var base = Math.pow(10, minLog + ratio * (maxLog - minLog));
      var jitterFactor = 1 + (rand() * 0.2 - 0.1);
      var jittered = base * jitterFactor;
      var rounded = Math.round(jittered * 100) / 100;
      LLs.push(rounded);
    }
    return LLs;
  }

  function getDiscountRange(delay) {
    if (delay >= 1 && delay <= 7) return { min: 5, max: 8 };
    else if (delay >= 8 && delay <= 30) return { min: 8, max: 13 };
    else if (delay >= 31 && delay <= 120) return { min: 13, max: 20 };
    else if (delay >= 121 && delay <= 365) return { min: 18, max: 28 };
    else return { min: 25, max: 40 };
  }

  // ---------------------------------------------------------------------------
  // Background selection: force ordered environments per discounting block
  // ---------------------------------------------------------------------------
  var ORDERED_LAYERS = ['layer1.png', 'layer2.png', 'layer3.png', 'layer4.png'];

  function getDirFromUrl(url) {
    if (!url || typeof url !== 'string') return null;
    // Remove query/hash
    var clean = url.split('#')[0].split('?')[0];
    var idx = clean.lastIndexOf('/');
    if (idx === -1) return null;
    return clean.substring(0, idx + 1);
  }

  function resolveOrderedLandUrl(blockIndex, maybeCurrentLandUrl) {
    var filename = ORDERED_LAYERS[blockIndex % ORDERED_LAYERS.length];
    // Best case: we can reuse the directory of whatever the bird task uses.
    var dir = getDirFromUrl(maybeCurrentLandUrl);
    if (dir) return dir + filename;

    // Fallback: if the bird code passes a relative "layerX.png" already, use that.
    // This is the least-assumptive fallback without touching other files.
    return filename;
  }

  // ---------------------------------------------------------------------------
  // Persistent background DOM (prevents flashing between trials)
  // ---------------------------------------------------------------------------
  // We keep a singleton background container in the jsPsych display element.
  // The background stays fixed for the whole discounting block; each trial just
  // updates the inner stage contents (#dd-stage).
  function ensureDiscountingFrame(display_element, skyUrl, landUrl) {
    var frame = display_element.querySelector('#dd-frame');
    var stage = display_element.querySelector('#dd-stage');

    // If frame exists, update background images ONLY if changed (avoid flashes).
    if (frame && stage) {
      var skyEl = frame.querySelector('.bg-sky');
      var landEl = frame.querySelector('.bg-land');

      if (skyEl && skyUrl) {
        // Only update if different
        if (skyEl.getAttribute('data-bg') !== skyUrl) {
          skyEl.style.backgroundImage = "url('" + skyUrl + "')";
          skyEl.setAttribute('data-bg', skyUrl);
        }
      }
      if (landEl && landUrl) {
        if (landEl.getAttribute('data-bg') !== landUrl) {
          landEl.style.backgroundImage = "url('" + landUrl + "')";
          landEl.setAttribute('data-bg', landUrl);
        }
      }
      return;
    }

    // Otherwise: create the full layered structure ONCE.
    // Clear body background so the container visuals are the only background.
    document.body.style.backgroundImage = '';
    document.body.style.backgroundSize = '';
    document.body.style.backgroundPosition = '';
    document.body.style.backgroundRepeat = '';

    var skyStyle = '';
    if (skyUrl) {
      skyStyle = "background-image: url('" + skyUrl + "'); background-size: cover; " +
        "background-position: center top; background-repeat: no-repeat;";
    }
    var landStyle = '';
    if (landUrl) {
      landStyle = "background-image: url('" + landUrl + "'); background-size: cover; " +
        "background-position: center bottom; background-repeat: no-repeat;";
    }

    display_element.innerHTML =
      '<div id="dd-frame" class="game-container">' +
        '<div class="bg-sky" data-bg="' + (skyUrl || '') + '" style="' + skyStyle + '"></div>' +
        '<div class="light1" id="light"></div>' +
        '<div class="bg-land" data-bg="' + (landUrl || '') + '" style="' + landStyle + '"></div>' +
        '<div class="main-container" style="position:relative;">' +
          // Stage is where we swap trial content without touching background layers.
          '<div id="dd-stage" style="position:absolute; top:50%; left:50%; ' +
          'transform:translate(-50%,-50%); width:100%; display:flex; justify-content:center;"></div>' +
        '</div>' +
      '</div>';
  }

  function setStageHTML(display_element, html) {
    var stage = display_element.querySelector('#dd-stage');
    if (stage) stage.innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Custom jsPsych plugin: delay-discounting (persistent background, retry logic)
  // ---------------------------------------------------------------------------
  jsPsych.plugins['delay-discounting'] = (function() {
    var plugin = {};

    plugin.info = {
      name: 'delay-discounting',
      description: 'Intertemporal choice between immediate and delayed rewards.',
      parameters: {
        SS_amount: { type: jsPsych.plugins.parameterType.FLOAT, default: undefined },
        LL_amount: { type: jsPsych.plugins.parameterType.FLOAT, default: undefined },
        delay_days: { type: jsPsych.plugins.parameterType.INT, default: undefined },
        discount_percent: { type: jsPsych.plugins.parameterType.FLOAT, default: undefined },
        bg_sky: { type: jsPsych.plugins.parameterType.STRING, default: null },
        bg_land: { type: jsPsych.plugins.parameterType.STRING, default: null }
      }
    };

    plugin.trial = function(display_element, trial) {
      var attempt = 1;
      var maxAttempts = 2;
      var responded = false;
      var choice = null;
      var timed_out = false;
      var skipped = false;
      var startTime = null;
      var choiceTimeoutID = null;

      // Lock the block background for this trial (captured in trial params).
      ensureDiscountingFrame(display_element, trial.bg_sky || null, trial.bg_land || null);

      function renderChoiceUI() {
        var html = '';
        html += '<div class="discount-container" style="display:flex; flex-direction:column; ' +
          'align-items:center; justify-content:center; text-align:center;">';
        html += '<p style="font-size:28px; margin-bottom:30px;">Choose the option you prefer</p>';
        html += '<div style="display:flex; flex-direction:row; justify-content:center; ' +
          'align-items:center; gap:200px; font-size:40px; font-weight:normal;">';
        html +=   '<div style="display:flex; flex-direction:column; align-items:center;">';
        html +=     '<div id="dd-left" style="border:2px solid #555; padding:20px 40px; ' +
          'border-radius:8px;">$' + Math.round(trial.SS_amount) +
          '<br><span style="font-size:20px;">now</span></div>';
        html +=     '<div style="font-size:24px; font-weight:bold; margin-top:10px; color:#333;">1</div>';
        html +=   '</div>';
        html +=   '<div style="display:flex; flex-direction:column; align-items:center;">';
        html +=     '<div id="dd-right" style="border:2px solid #555; padding:20px 40px; ' +
          'border-radius:8px;">$' + Math.round(trial.LL_amount) +
          '<br><span style="font-size:20px;">in ' + trial.delay_days + ' days</span></div>';
        html +=     '<div style="font-size:24px; font-weight:bold; margin-top:10px; color:#333;">2</div>';
        html +=   '</div>';
        html += '</div>';
        html += '<p style="font-size:16px; margin-top:20px; color:#555555;">Press 1 or 2 to select an option.</p>';
        html += '</div>';
        setStageHTML(display_element, html);

        // Wire keyboard input (1 and 2 keys)
        function keyboardHandler(e) {
          if (e.key === '1') {
            e.preventDefault();
            handleChoice('left');
          } else if (e.key === '2') {
            e.preventDefault();
            handleChoice('right');
          }
        }
        document.addEventListener('keydown', keyboardHandler);

        // Store handler for cleanup
        window._ddKeyboardHandler = keyboardHandler;

        // Start timer
        responded = false;
        startTime = performance.now();
        choiceTimeoutID = setTimeout(handleTimeout, 10000);
      }

      function handleChoice(side) {
        if (responded) return;
        responded = true;
        clearTimeout(choiceTimeoutID);

        var rt = Math.round(performance.now() - startTime);
        choice = (side === 'left') ? 'SS' : 'LL';
        timed_out = false;
        skipped = false;

        // Highlight selection
        var leftEl = display_element.querySelector('#dd-left');
        var rightEl = display_element.querySelector('#dd-right');
        if (side === 'left') {
          leftEl.style.fontWeight = 'bold';
          leftEl.style.color = '#007bff';
        } else {
          rightEl.style.fontWeight = 'bold';
          rightEl.style.color = '#007bff';
        }

        // IMPORTANT: do NOT clear the entire display_element (keeps background)
        setTimeout(function() {
          setStageHTML(display_element, ''); // ITI
          setTimeout(function() {
            finishTrial(rt);
          }, 250);
        }, 500);
      }

      function handleTimeout() {
        if (responded) return;
        responded = true;
        clearTimeout(choiceTimeoutID);

        var rt = 10000;

        // Show timeout message in the stage (background remains)
        var html = '';
        html += '<div class="discount-container" style="display:flex; flex-direction:column; ' +
          'align-items:center; justify-content:center; text-align:center; border:4px solid red; ' +
          'padding:10px 20px; border-radius:8px;">';
        html += '<p style="font-size:28px; margin-bottom:20px;">Choose the option you prefer</p>';
        html += '<p style="font-size:24px; margin-top:10px; color:#d9534f;">' +
          'No selection made. Please make a selection within the allowed time.</p>';
        html += '</div>';
        setStageHTML(display_element, html);

        setTimeout(function() {
          if (attempt < maxAttempts) {
            attempt += 1;
            renderChoiceUI(); // retry SAME parameters, SAME background, no flash
          } else {
            timed_out = true;
            skipped = true;
            choice = 'NA';
            setStageHTML(display_element, '');
            finishTrial(rt);
          }
        }, 5000);
      }

      function finishTrial(rt) {
        clearTimeout(choiceTimeoutID);

        // Remove keyboard handler
        if (window._ddKeyboardHandler) {
          document.removeEventListener('keydown', window._ddKeyboardHandler);
          delete window._ddKeyboardHandler;
        }

        var trial_data = {};
        if (typeof trial.data === 'object') {
          for (var prop in trial.data) trial_data[prop] = trial.data[prop];
        }

        trial_data.choice = choice || 'NA';

        // Analysis-friendly binary coding: LL=1, SS=0, missing=null
        trial_data.choice_LL = (trial_data.choice === 'LL') ? 1 : (trial_data.choice === 'SS') ? 0 : null;
        trial_data.reaction_time = rt;
        trial_data.attempt_number = attempt;
        trial_data.timed_out = timed_out;
        trial_data.skipped = skipped;

        // Store which background was actually used (for debugging)
        trial_data.bg_sky = trial.bg_sky || null;
        trial_data.bg_land = trial.bg_land || null;

        jsPsych.finishTrial(trial_data);
      }

      // First render
      renderChoiceUI();
    };

    return plugin;
  })();

  // ---------------------------------------------------------------------------
  // Custom jsPsych plugin: delay-discounting-instructions (with persistent background)
  // ---------------------------------------------------------------------------
  jsPsych.plugins['delay-discounting-instructions'] = (function() {
    var plugin = {};

    plugin.info = {
      name: 'delay-discounting-instructions',
      description: 'Instructions with persistent discounting background.',
      parameters: {
        pages: { type: jsPsych.plugins.parameterType.COMPLEX, default: [] },
        bg_sky: { type: jsPsych.plugins.parameterType.STRING, default: null },
        bg_land: { type: jsPsych.plugins.parameterType.STRING, default: null }
      }
    };

    plugin.trial = function(display_element, trial) {
      ensureDiscountingFrame(display_element, trial.bg_sky || null, trial.bg_land || null);

      var html = '<div style="display:flex; flex-direction:column; align-items:center; ' +
        'justify-content:center; text-align:center;">';
      html += '<p style="font-size:28px;">' + trial.pages[0] + '</p>';
      html += '<button id="dd-instr-continue" style="margin-top:20px; padding:10px 20px; ' +
        'font-size:20px;">Continue</button>';
      html += '</div>';
      setStageHTML(display_element, html);

      var continueButton = document.getElementById('dd-instr-continue');
      var autoAdvanceTimeout = setTimeout(function() {
        setStageHTML(display_element, '');
        jsPsych.finishTrial({});
      }, 10000);

      continueButton.addEventListener('click', function() {
        clearTimeout(autoAdvanceTimeout);
        setStageHTML(display_element, '');
        jsPsych.finishTrial({});
      });
    };

    return plugin;
  })();

  // ---------------------------------------------------------------------------
  // Timeline builder: runDelayDiscounting(conditionTag)
  // ---------------------------------------------------------------------------
  async function runDelayDiscounting(conditionTag) {
    var pid = jsPsych.data.getURLVariable('participant_id') ||
              jsPsych.data.getURLVariable('PROLIFIC_PID') ||
              jsPsych.data.getURLVariable('workerId') ||
              jsPsych.data.getURLVariable('subject_id');

    if (!pid) {
      try {
        var first = jsPsych.data.get().first(1).values()[0];
        if (first && typeof first.participant_id !== 'undefined') pid = first.participant_id;
      } catch (ex) {}
    }
    if (!pid) pid = 'NA';

    // Parse conditionTag like "vol4_stc64" into numeric parameters for analysis
    var volParam = null;
    var stcParam = null;
    if (typeof conditionTag === 'string') {
      var m = conditionTag.match(/vol(\d+)_stc(\d+)/i);
      if (m) {
        volParam = parseInt(m[1], 10);
        stcParam = parseInt(m[2], 10);
      }
    }


    // IMPORTANT: Use current block counter BEFORE increment to pick the fixed environment order.
    var blockIndex = delayBlockCounter % 4;

    // Capture sky/backgrounds for this block ONCE and embed into every trial.
    var skyUrl = (typeof window !== 'undefined' && window.sky_background) ? window.sky_background : null;

    // Force land to match the deterministic environment order. If we have a land URL from the bird block,
    // use its directory to construct layerX.png so paths stay consistent with your project.
    var maybeLand = (typeof window !== 'undefined' && window.current_land_img) ? window.current_land_img : null;
    var landUrl = resolveOrderedLandUrl(blockIndex, maybeLand);

    // Seed RNG
    var baseSeed = stringToSeed(pid);
    var tagSeed = stringToSeed(conditionTag || '');
    var localSeed = (baseSeed + tagSeed + delayBlockCounter) >>> 0;

    // Increment counter once per block
    delayBlockCounter += 1;

    var rand = mulberry32(localSeed);

    var numMain = 12;
    var delays = generateDelays(rand, numMain);

    var randLL = mulberry32((localSeed + 12345) >>> 0);
    var LLs = generateLLs(randLL, numMain);

    var randPair = mulberry32((localSeed + 67890) >>> 0);
    for (var i = LLs.length - 1; i > 0; i--) {
      var j = Math.floor(randPair() * (i + 1));
      var tmp = LLs[i];
      LLs[i] = LLs[j];
      LLs[j] = tmp;
    }

    var timeline = [];

    timeline.push({
      type: 'delay-discounting-instructions',
      pages: [
        'You will now see a series of hypothetical choices. Each choice will consist ' +
        'of two monetary rewards, received at different points in time. <br><br>Chose ' +
        'the option that you would most prefer to receive by pressing 1 or 2 on your keyboard.'
      ],
      bg_sky: skyUrl,
      bg_land: landUrl
    });

    // Build main trials
    var mainTrials = [];
    for (var t = 0; t < numMain; t++) {
      var delayDays = delays[t];
      var LL = LLs[t];
      var range = getDiscountRange(delayDays);
      var p = range.min + randPair() * (range.max - range.min);

      var SS = LL * (1 - p / 100);
      var maxBeforeRound = LL * 0.95 - 0.01;
      if (SS >= maxBeforeRound) SS = maxBeforeRound;

      SS = Math.round(SS);
      LL = Math.round(LL);

      var maxSSRounded = Math.floor(LL * 0.95);
      if (SS >= maxSSRounded) SS = maxSSRounded - 1;
      if (SS < 1) SS = 1;

      mainTrials.push({
        type: 'delay-discounting',
        SS_amount: SS,
        LL_amount: LL,
        delay_days: delayDays,
        discount_percent: p,
        bg_sky: skyUrl,
        bg_land: landUrl,
        data: {
          participant_id: pid,
          condition: conditionTag,
        true_vol_param: volParam,
        true_stc_param: stcParam,
          true_vol_param: volParam,
          true_stc_param: stcParam,
          SS_amount: SS,
          LL_amount: LL,
          delay_days: delayDays,
          discount_percent: p,
          attention_check: false
        }
      });
    }

    // Attention checks: 4 SS-dominated specs (1 per block, log-distributed) + 1 random LL-dominated per block
    // Helper: evenly log-distribute values between start and end
    function logSpace(start, end, num) {
      var arr = [];
      var minLog = Math.log10(start);
      var maxLog = Math.log10(end);
      for (var i = 0; i < num; i++) {
        var ratio = (num === 1) ? 0 : i / (num - 1);
        arr.push(Math.pow(10, minLog + ratio * (maxLog - minLog)));
      }
      return arr;
    }

    // 4 blocks, so 4 log-distributed baseSS values between 5 and 95
    var numBlocks = 4;
    var baseSSs = logSpace(5, 95, numBlocks).map(function(x) { return Math.round(x); });
    var baseDelays = [1, 10, 100, 950]; // Fixed delays for each SS-dominated attention check

    var allSSSpecs = [];
    for (var i = 0; i < numBlocks; i++) {
      allSSSpecs.push({ baseSS: baseSSs[i], baseDelay: baseDelays[i] });
    }

    // Track which SS specs have been used across blocks
    if (typeof window.usedSSIndices === 'undefined') {
      window.usedSSIndices = [];
    }

    // Select 1 random SS spec that hasn't been used yet
    var availableSSIndices = [];
    for (var i = 0; i < allSSSpecs.length; i++) {
      if (window.usedSSIndices.indexOf(i) === -1) {
        availableSSIndices.push(i);
      }
    }

    // Randomly pick 1 from available
    var randACPick = mulberry32((localSeed + 98765) >>> 0);
    var selectedSSIndices = [];
    if (availableSSIndices.length >= 1) {
      var pickIdx = Math.floor(randACPick() * availableSSIndices.length);
      selectedSSIndices.push(availableSSIndices[pickIdx]);
      window.usedSSIndices.push(selectedSSIndices[0]);
    } else {
      // Fallback if we run out (shouldn't happen with only 4 blocks)
      selectedSSIndices = [0];
    }

    // Generate 1 random LL-dominated spec (truly random, not seeded)
    var attentionTrials = [];

    // Add 1 SS-dominated check
    for (var ss_idx = 0; ss_idx < selectedSSIndices.length; ss_idx++) {
      var spec = allSSSpecs[selectedSSIndices[ss_idx]];

      var SS_j = Math.round(spec.baseSS * (1 + (randACPick() * 0.2 - 0.1)));
      if (SS_j < 1) SS_j = 1;
      if (SS_j > 100) SS_j = 100;

      var delay_j = Math.round(spec.baseDelay * (1 + (randACPick() * 0.2 - 0.1)));
      if (delay_j < 1) delay_j = 1;
      if (delay_j > 1000) delay_j = 1000;

      var drange = getDiscountRange(delay_j);
      var p_ac = drange.min + randACPick() * (drange.max - drange.min);

      // SS-dominated: LL < SS
      var LL_ac = Math.round(SS_j * (1 - p_ac / 100));
      if (LL_ac >= SS_j) LL_ac = SS_j - 1;
      if (LL_ac < 1) LL_ac = 1;

      attentionTrials.push({
        type: 'delay-discounting',
        SS_amount: SS_j,
        LL_amount: LL_ac,
        delay_days: delay_j,
        discount_percent: p_ac,
        bg_sky: skyUrl,
        bg_land: landUrl,
        data: {
          participant_id: pid,
          condition: conditionTag,
        true_vol_param: volParam,
        true_stc_param: stcParam,
          SS_amount: SS_j,
          LL_amount: LL_ac,
          delay_days: delay_j,
          discount_percent: p_ac,
          attention_check: true,
          ac_type: 'SS_dominated'
        }
      });
    }

    // Add 1 LL-dominated check (truly random using Math.random())
    var SS_ll = 0; // LL-dominated: SS is always $0
    var LL_ll = Math.round(Math.random() * 100); // LL random $0-100
    if (LL_ll < 1) LL_ll = 1; // Ensure at least $1

    var delay_ll = Math.round(Math.random() * 1000); // Delay random 0-1000 days
    if (delay_ll < 1) delay_ll = 1;

    attentionTrials.push({
      type: 'delay-discounting',
      SS_amount: SS_ll,
      LL_amount: LL_ll,
      delay_days: delay_ll,
      discount_percent: 100, // Always 100% discount for LL-dominated
      bg_sky: skyUrl,
      bg_land: landUrl,
      data: {
        participant_id: pid,
        condition: conditionTag,
        SS_amount: SS_ll,
        LL_amount: LL_ll,
        delay_days: delay_ll,
        discount_percent: 100,
        attention_check: true,
        ac_type: 'LL_dominated'
      }
    });

    // Randomly shuffle the 2 attention trials
    var randACOrder = mulberry32((localSeed + 99999) >>> 0);
    for (var a = attentionTrials.length - 1; a > 0; a--) {
      var k = Math.floor(randACOrder() * (a + 1));
      var temp = attentionTrials[a];
      attentionTrials[a] = attentionTrials[k];
      attentionTrials[k] = temp;
    }

    var acPositions = [5, 10];
    var finalTrials = [];
    var mainIndex = 0;
    var acIndex = 0;

    for (var n = 1; n <= numMain + 2; n++) {
      if (acPositions.indexOf(n) !== -1) {
        var acTrial = attentionTrials[acIndex++];
        acTrial.data.trial_index = n;
        acTrial.data.dd_trial = n;
        finalTrials.push(acTrial);
      } else {
        var mTrial = mainTrials[mainIndex++];
        mTrial.data.trial_index = n;
        mTrial.data.dd_trial = n;
        finalTrials.push(mTrial);
      }
    }

    for (var ft = 0; ft < finalTrials.length; ft++) timeline.push(finalTrials[ft]);

    // End screen (instructions plugin, already loaded in your pipeline)
    timeline.push({
      type: 'html-keyboard-response',
      stimulus: `
        <p style="font-size:28px; text-align:center;">
          You have completed this choice section. You will now move on to the memory task.<br><br>
          Press any arrow key or space bar to move on (auto in 10s).
        </p>
      `,
      choices: [32, 37, 38, 39, 40], // space, left, up, right, down
      trial_duration: 10000,
    });

    return timeline;
  }

  window.runDelayDiscounting = runDelayDiscounting;
})();