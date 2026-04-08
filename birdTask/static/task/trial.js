jsPsych.plugins["trial"] = (function () {
  var plugin = {};

  plugin.info = {
    name: "trial",
    description: "",
    parameters: {
      dummy: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "dummy",
        default: null,
        description: "starting location"
      },
      terminate_now: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false
      },
      show_missing: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false
      },
      show_drone: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false
      },
      show_bird: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false
      },
      is_moving_practice: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false
      },
      canvas_size: {
        type: jsPsych.plugins.parameterType.INT,
        array: true,
        pretty_name: "Canvas size",
        default: [2000, 2000],
        description: "Array containing the height (first value) and width (second value) of the canvas element."
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: "Choices",
        default: [32, 37, 39],
        description: "Keys corresponding to each context (left, right, down)."
      },
      strong_warning: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false
      },
      missing_msg_warning_number: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: 15
      },
      bucket_position: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: null,
        description: "location of the collector."
      },
      stayed: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 1,
        description: "0 if the collector moved this trial, otherwise 1."
      },
      coins_distribution: {
        type: jsPsych.plugins.parameterType.INT,
        array: true,
        pretty_name: "distribution of drop objects",
        default: [-4.25, -3, -1.75, -0.75, -0.25, 0.25, 0.75, 1.75, 3, 4.25],
        description: "distribution of drop objects."
      },
      coins_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: true,
        default: [350, 400, 500, 550, 600, 600, 600, 500, 500, 400],
        description: "falling duration of drop objects."
      },
      bag_position: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        pretty_name: "x of drop bag",
        default: 35,
        description: "x of drop bag"
      },
      bird_position: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: null,
        description: "x of the drone (legacy name kept for data compatibility)"
      },
      no_response_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 1500
      },
      response_remaining_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 2500
      },
      drop_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 5000
      },
      missing_duration: {
        type: jsPsych.plugins.parameterType.INT,
        array: false,
        default: 7500
      },
      stim_img: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null,
        description: "Path to the stimulus image for the memory component."
      },
      background_img: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null,
        description: "Path to the background image used to cue the block context."
      },
      land_img: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null,
        description: "Path to the land background image for this block."
      },
      hide_stimulus: {
        type: jsPsych.plugins.parameterType.BOOL,
        array: false,
        default: false,
        description: "If true, keep the stimulus hidden even after the drop objects land."
      },
      valence: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: "reward",
        description: "'reward' or 'loss' — controls visual mode for this trial."
      },
      vol_level: {
        type: jsPsych.plugins.parameterType.STRING,
        array: false,
        default: null
      }
    }
  };

  var PARTICLE_COUNT = 10;
  var DROP_OBJ_COUNT = 10;
  var _explTimeouts = [];

  function getLandImg(trial) {
    if (trial.land_img) return trial.land_img;

    var valence = trial.valence || "reward";
    var vol = trial.vol_level || "low";

    if (valence === "loss") {
      return vol === "high"
        ? "/static/img/task_assets/loss/layer_loss_highVol.png"
        : "/static/img/task_assets/loss/layer_loss_lowVol.png";
    }

    return vol === "high"
      ? "/static/img/task_assets/reward/layer_reward_highVol.png"
      : "/static/img/task_assets/reward/layer_reward_lowVol.png";
  }

  function getValenceAssets(valence) {
    if (typeof VALENCE_CONFIG !== "undefined" && VALENCE_CONFIG[valence]) {
      return VALENCE_CONFIG[valence];
    }

    if (valence === "loss") {
      return {
        drone_img: "/static/img/task_assets/loss/drone0.png",
        bag_img: "/static/img/task_assets/loss/hazard-bag.png",
        dot_img: "/static/img/task_assets/loss/hazard-dot.png",
        lock_overlay_bg: "rgba(35, 0, 55, 0.30)",
        lock_overlay_opacity: "1",
        feedback_sign: function (c) { return c - 10; },
        feedback_color: function (c) {
          var v = c - 10;
          return v === 0 ? "#ffd700" : "#ff4444";
        }
      };
    }

    return {
      drone_img: "/static/img/task_assets/reward/drone0.png",
      bag_img: "/static/img/task_assets/reward/supply-bag.png",
      dot_img: "/static/img/task_assets/reward/supply-dot.png",
      lock_overlay_bg: "rgba(0, 0, 0, 0.24)",
      lock_overlay_opacity: "1",
      feedback_sign: function (c) { return c; },
      feedback_color: function (c) {
        return c === 10 ? "#39FF14" : (c === 0 ? "#ffd700" : "#39FF14");
      }
    };
  }

  function make_html(trial) {
    var new_html = "";
    var valence = trial.valence || "reward";
    var va = getValenceAssets(valence);
    var landSrc = getLandImg(trial);

    trial.dummy = trial.coins_distribution.length;

    new_html += '<style>\n' +
      'body { height:100vh; max-height:100vh; overflow:hidden; position:fixed; background-color:#000; }\n' +
      '.explosion-container { position:absolute; width:0; height:0; z-index:16; pointer-events:none; }\n' +
      '.expl-flash { position:absolute; top:50%; left:50%; width:clamp(22px,4.5vw,55px); height:clamp(22px,4.5vw,55px); border-radius:50%; transform:translate(-50%,-50%) scale(0.3); opacity:0; pointer-events:none; }\n' +
      '.expl-flash.fire { animation: flashPop 220ms ease-out forwards; }\n' +
      '.expl-flash.reward { background:radial-gradient(circle, rgba(255,248,190,.95), rgba(255,215,0,.45) 55%, transparent 100%); }\n' +
      '.expl-flash.loss { background:radial-gradient(circle, rgba(255,210,210,.95), rgba(255,68,68,.45) 55%, transparent 100%); }\n' +
      '@keyframes flashPop { 0%{transform:translate(-50%,-50%) scale(0.3);opacity:1} 35%{transform:translate(-50%,-50%) scale(1.15);opacity:.85} 100%{transform:translate(-50%,-50%) scale(1.5);opacity:0} }\n' +
      '.expl-ring { position:absolute; top:50%; left:50%; width:0; height:0; border-radius:50%; transform:translate(-50%,-50%); opacity:0; pointer-events:none; }\n' +
      '.expl-ring.fire { animation:ringExpand 400ms cubic-bezier(.22,.61,.36,1) forwards; }\n' +
      '.expl-ring.reward { border:2.5px solid rgba(255,215,0,.65); box-shadow:0 0 16px 4px rgba(255,215,0,.25), inset 0 0 10px rgba(255,215,0,.12); }\n' +
      '.expl-ring.loss { border:2.5px solid rgba(255,68,68,.65); box-shadow:0 0 16px 4px rgba(255,68,68,.25), inset 0 0 10px rgba(255,68,68,.12); }\n' +
      '@keyframes ringExpand { 0%{width:0;height:0;opacity:.9} 45%{opacity:.75} 100%{width:clamp(65px,10vw,130px);height:clamp(65px,10vw,130px);opacity:0} }\n' +
      '.expl-particle { position:absolute; top:50%; left:50%; border-radius:50%; opacity:0; pointer-events:none; transform:translate(-50%,-50%); }\n' +
      '.expl-particle.reward { background:radial-gradient(circle,#fffbe0,#ffd700); box-shadow:0 0 4px rgba(255,215,0,.5); }\n' +
      '.expl-particle.loss { background:radial-gradient(circle,#ffd4d4,#ff4444); box-shadow:0 0 4px rgba(255,68,68,.5); }\n' +
      '@keyframes objDropSpread { 0%{top:72%;transform:translateX(0) scale(0.8)} 15%{top:68%;transform:translateX(calc(var(--spread-x)*0.2)) scale(1.1)} 100%{top:105%;transform:translateX(var(--spread-x)) scale(1)} }\n' +
      '</style>\n';

    new_html += '<div class="game-container">';
    new_html += '<div class="bg-sky"></div>';
    new_html += '<div class="bg-land" style="background-image:url(\'' + landSrc + '\');background-size:cover;background-position:center bottom;background-repeat:no-repeat;"></div>';
    new_html += '<div class="main-container">';

    new_html += '<div class="collector-wrap" id="collector" style="left:' + trial.bucket_position + '%;">' +
  '<img class="collector-base" id="collector-base" src="/static/img/task_assets/shared/bucket1.png" alt="collector" />' +
  '<img class="collector-glow" id="collector-glow" src="/static/img/task_assets/shared/bucket2.png" alt="" style="opacity:.25;" />' +
  '<div id="collector-lock-shade" style="position:absolute;inset:0;border-radius:10px;background:rgba(0,0,0,0);pointer-events:none;transition:background .2s ease, opacity .2s ease;opacity:0;"></div>' +
  '</div>';

    new_html += '<img class="asset drone-el" id="drone" src="' + (va.drone_img || "") + '" alt="drone" style="visibility:hidden;" />';
    new_html += '<img class="asset drop-bag" id="drop-bag" src="' + (va.bag_img || "") + '" alt="bag" style="opacity:0; animation:none; top:18%;" />';

    new_html += '<div class="explosion-container" id="expl-container">' +
      '<div class="expl-flash ' + valence + '" id="expl-flash"></div>' +
      '<div class="expl-ring ' + valence + '" id="expl-ring"></div>' +
      '</div>';

    for (var i = 0; i < DROP_OBJ_COUNT; i++) {
      new_html += '<img class="drop-obj" id="drop-obj' + i + '" src="' + (va.dot_img || "") + '" alt="" style="visibility:hidden;" />';
    }

    new_html += '<img src="' + (trial.stim_img || "") + '" class="stimulus-img" id="stimulus-img" style="visibility:hidden; position:absolute;" />';
    new_html += '<div class="fb-overlay" id="fb" style="opacity:0;"></div>';

    new_html += "</div>";
    new_html += "</div>";
    return new_html;
  }

  function triggerExplosion(leftPercent, topPercent, mode) {
    var container = document.getElementById("expl-container");
    if (!container) return;

    container.style.left = leftPercent + "%";
    container.style.top = topPercent;

    var oldParticles = container.querySelectorAll(".expl-particle");
    for (var op = 0; op < oldParticles.length; op++) oldParticles[op].remove();

    var flash = document.getElementById("expl-flash");
    if (flash) {
      flash.className = "expl-flash " + mode;
      void flash.offsetWidth;
      flash.classList.add("fire");
    }

    var ring = document.getElementById("expl-ring");
    if (ring) {
      ring.className = "expl-ring " + mode;
      void ring.offsetWidth;
      ring.classList.add("fire");
    }

    for (var i = 0; i < PARTICLE_COUNT; i++) {
      var p = document.createElement("div");
      p.className = "expl-particle " + mode;

      var baseAngle = (2 * Math.PI / PARTICLE_COUNT) * i;
      var jitter = (Math.random() - 0.5) * 0.55;
      var angle = baseAngle + jitter;
      var dist = 22 + Math.random() * 35;
      var dur = 260 + Math.random() * 200;
      var size = 3 + Math.random() * 4;
      var delay = Math.random() * 35;

      p.style.width = size + "px";
      p.style.height = size + "px";
      p.style.transition = "none";
      p.style.opacity = "1";
      p.style.transform = "translate(-50%,-50%) translate(0px,0px) scale(1)";

      container.appendChild(p);
      void p.offsetWidth;

      var dx = Math.cos(angle) * dist;
      var dy = Math.sin(angle) * dist;

      (function (el, _dx, _dy, _dur, _delay) {
        requestAnimationFrame(function () {
          el.style.transition = "transform " + _dur + "ms cubic-bezier(.15,.75,.3,1), opacity " + _dur + "ms ease-out";
          el.style.transitionDelay = _delay + "ms";
          el.style.transform = "translate(-50%,-50%) translate(" + _dx + "px," + _dy + "px) scale(0.25)";
          el.style.opacity = "0";
        });
        var tid = setTimeout(function () { if (el.parentNode) el.remove(); }, _dur + _delay + 60);
        _explTimeouts.push(tid);
      })(p, dx, dy, dur, delay);
    }

    var resetTid = setTimeout(function () {
      if (flash) flash.classList.remove("fire");
      if (ring) ring.classList.remove("fire");
    }, 480);
    _explTimeouts.push(resetTid);
  }

  function move_collector(trial, info) {
    var key = info.key;
    var x = trial.bucket_position;
    var step = 2;

    if (key == 39) x = x + step;
    else if (key == 37) x = x - step;

    if (x > 90) x = 90;
    else if (x < 10) x = 10;

    var collector = document.getElementById("collector");
    if (collector) collector.style.left = x + "%";

    trial.bucket_position = x;
    return trial;
  }

  function drop_objects(trial) {
    var durations = trial.coins_duration;
    for (var i = 0; i < trial.coins_distribution.length; i++) {
      var obj = document.getElementById("drop-obj" + i);
      if (!obj) continue;
      obj.style.visibility = "visible";
      obj.style.left = (trial.bag_position + trial.coins_distribution[i]) + "%";
      obj.style.top = "72%";
      obj.style.setProperty("--spread-x", (trial.coins_distribution[i] * 0.5) + "vw");
      obj.style.animation = "objDropSpread " + durations[i] + "ms cubic-bezier(0.4,0,1,1) forwards";
    }

    var glow = document.getElementById("collector-glow");
    if (glow) glow.style.opacity = "1";
  }

  function fly(trial) {
    var valence = trial.valence || "reward";
    var va = getValenceAssets(valence);

    var initial_duration = 300;
    var bag_duration = 1000;
    var gold_initialization = 200;
    var explode_duration = bag_duration + gold_initialization;
    var captureIndices = [];

    var collectorBase = document.getElementById("collector-base");
    var collectorGlow = document.getElementById("collector-glow");
    var collectorShade = document.getElementById("collector-lock-shade");

    if (collectorBase) {
      collectorBase.style.filter = "brightness(0.72) saturate(0.9)";
    }
    if (collectorGlow) {
      collectorGlow.style.opacity = "0.85";
    }
    if (collectorShade) {
      collectorShade.style.background = (valence === "loss")
        ? "rgba(60,0,80,0.22)"
        : "rgba(0,0,0,0.22)";
      collectorShade.style.opacity = "1";
    }

    setTimeout(function () {
      if (trial.show_drone || trial.show_bird) {
        var drone = document.getElementById("drone");
        if (drone) {
          drone.style.left = trial.bird_position + "%";
          drone.style.visibility = "visible";
        }
      }

      var bag = document.getElementById("drop-bag");
      if (bag) {
        bag.style.left = trial.bag_position + "%";
        bag.style.opacity = "1";
        bag.style.top = "18%";
        bag.style.animation = "bagdrop " + bag_duration + "ms ease-in forwards";
      }

      var stim = document.getElementById("stimulus-img");
      if (stim) {
        stim.style.visibility = "hidden";
        stim.style.left = trial.bag_position + "%";
        stim.style.top = "50%";
      }
    }, initial_duration);

    setTimeout(function () {
      var bag = document.getElementById("drop-bag");
      if (bag) {
        bag.style.opacity = "0";
        bag.style.animation = "none";
      }

      triggerExplosion(trial.bag_position, "82%", valence);
      drop_objects(trial);

      for (var ci = 0; ci < trial.coins_distribution.length; ci++) {
        (function (idx) {
          var dur = trial.coins_duration[idx];
          var capFrac = 19 / 30;

          setTimeout(function () {
            var objEl = document.getElementById("drop-obj" + idx);
            var collEl = document.getElementById("collector");
            if (!objEl || !collEl) return;

            var or = objEl.getBoundingClientRect();
            var br = collEl.getBoundingClientRect();
            var ow = or.width || 8;
            var overlap = Math.max(0, Math.min(or.right, br.right) - Math.max(or.left, br.left));

            if (overlap >= ow) {
              captureIndices.push(idx);
              objEl.style.visibility = "hidden";
            }
          }, dur * capFrac);

          setTimeout(function () {
            var objEl = document.getElementById("drop-obj" + idx);
            if (objEl) objEl.style.visibility = "hidden";
          }, dur + 50);
        })(ci);
      }
    }, initial_duration + explode_duration);

    var maxDur = 0;
    for (var ii = 0; ii < trial.coins_duration.length; ii++) {
      if (trial.coins_duration[ii] > maxDur) maxDur = trial.coins_duration[ii];
    }

    setTimeout(function () {
      var captureCount = captureIndices.length;
      trial.coins_caught = captureCount;

      var valueChange = va.feedback_sign(captureCount);
      var displayText;
      if (valueChange > 0) displayText = "+" + valueChange;
      else if (valueChange < 0) displayText = String(valueChange);
      else displayText = "0";

      var color = va.feedback_color(captureCount);

      var fb = document.getElementById('fb');
      if (fb) {
        fb.textContent = displayText;
        fb.style.color = color;
        fb.style.left = trial.bag_position + "%";
        fb.style.top = "86%";
        fb.style.transform = "translate(-50%,-100%)";
        fb.style.opacity = "1";
      }

      var stimEl = document.getElementById("stimulus-img");
      if (stimEl && trial.stim_img) {
        stimEl.style.left = trial.bag_position + "%";
        stimEl.style.top = "78%";
        stimEl.style.transform = "translate(-50%,-100%)";
        stimEl.style.width = "40%";
        stimEl.style.height = "auto";
        if (!trial.hide_stimulus) {
          stimEl.style.visibility = "visible";
        }
      }

      setTimeout(function () {
        if (fb) fb.style.opacity = "0";
        if (stimEl) stimEl.style.visibility = "hidden";
      }, 2500);

    }, initial_duration + explode_duration + maxDur + 100);
  }

  plugin.trial = function (display_element, trial) {
    var _trial_onset_perf = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
    var _trial_onset_elapsed = (typeof jsPsych !== "undefined" && jsPsych.totalTime) ? jsPsych.totalTime() : null;

    var _bucket_start_pos = trial.bucket_position;
    var _bucket_end_pos = trial.bucket_position;
    var _rt_first_move_ms = null;
    var _rt_last_move_ms = null;
    var _num_moves = 0;

    var _record_move_if_changed = function (prev_pos) {
      if (trial.bucket_position !== prev_pos) {
        var now = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
        var ms = now - _trial_onset_perf;
        if (_rt_first_move_ms === null) _rt_first_move_ms = ms;
        _rt_last_move_ms = ms;
        _num_moves += 1;
        _bucket_end_pos = trial.bucket_position;
      }
    };

    if (trial.show_bird && !trial.show_drone) trial.show_drone = trial.show_bird;

    if (trial.terminate_now) {
      setTimeout(function () { end_trial(2); }, 1);
    }

    if (trial.is_moving_practice) {
      trial.no_response_duration = 5000;
    }

    display_element.innerHTML = make_html(trial);

    var collector = document.getElementById("collector");
    if (collector) collector.style.left = trial.bucket_position + "%";

    var after_response = function (info) {
      var prev_pos = trial.bucket_position;
      trial = move_collector(trial, info);
      _record_move_if_changed(prev_pos);
    };

    var after_1st_response = function (info) {
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener1);
      trial.stayed = 0;

      var prev_pos = trial.bucket_position;
      trial = move_collector(trial, info);
      _record_move_if_changed(prev_pos);

      keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: "performance",
        persist: true,
        allow_held_key: true
      });

      setTimeout(function () {
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
        if (!trial.is_moving_practice) {
          fly(trial);
        }
        jsPsych.pluginAPI.setTimeout(function () {
          end_trial(1);
        }, trial.drop_duration);
      }, trial.response_remaining_duration);
    };

    var keyboardListener1 = null;
    var keyboardListener = null;

    setTimeout(function () {
      keyboardListener1 = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_1st_response,
        valid_responses: trial.choices,
        rt_method: "performance",
        persist: false,
        allow_held_key: true
      });
    }, 20);

    var missed_response = function () {
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      var msg;

      if (trial.is_moving_practice) {
        msg = '<p style="font-size: 20px; line-height: 1.5em">Are you there? You should try to move the collector using left and right arrow keys.<br><br>Please pay more attention and move the collector, otherwise we will end the game here!</p>';
      } else if (trial.strong_warning) {
        msg = '<p style="font-size: 20px; line-height: 1.5em">Are you there? You have not moved the collector for a long time.<br><br>We have warned you more than ' + trial.missing_msg_warning_number + ' times. <br><br> <b>Warning: we are about to reject your work!</b></p>';
      } else {
        msg = '<p style="font-size: 20px; line-height: 1.5em">Are you there? You have not moved the collector for a long time.<br><br>Please pay more attention and play with your collector, otherwise we may end the experiment early and reject your work.</p>';
      }

      display_element.innerHTML = msg;
      jsPsych.pluginAPI.setTimeout(function () { end_trial(0); }, trial.missing_duration);
    };

    var end_trial = function (completed) {
      jsPsych.pluginAPI.clearAllTimeouts();
      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      for (var et = 0; et < _explTimeouts.length; et++) clearTimeout(_explTimeouts[et]);
      _explTimeouts = [];

    var collectorBase = document.getElementById("collector-base");
    var collectorGlow = document.getElementById("collector-glow");
    var collectorShade = document.getElementById("collector-lock-shade");

    if (collectorBase) collectorBase.style.filter = "brightness(1) saturate(1)";
    if (collectorGlow) collectorGlow.style.opacity = ".25";
    if (collectorShade) {
      collectorShade.style.opacity = "0";
      collectorShade.style.background = "rgba(0,0,0,0)";
    }

      var coins_caught_value = (completed === 1) ? trial.coins_caught : null;

      var _trial_duration_ms = null;
      try {
        var now = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
        _trial_duration_ms = now - _trial_onset_perf;
      } catch (e) {}

      var movement_duration_ms = (_rt_first_move_ms !== null && _rt_last_move_ms !== null)
        ? (_rt_last_move_ms - _rt_first_move_ms) : null;

      var miss_reason = null;
      if (completed === 0) miss_reason = "no_move_timeout";
      if (completed === 2) miss_reason = "terminated";

      var trial_data = {
        bird_position: trial.bird_position,
        bag_position: trial.bag_position,
        bucket_position: trial.bucket_position,
        completed: completed,
        stayed: trial.stayed,
        coins_caught: coins_caught_value,
        valence: trial.valence || "reward",
        bucket_start_pos: _bucket_start_pos,
        bucket_end_pos: _bucket_end_pos,
        num_moves: _num_moves,
        rt_first_move_ms: _rt_first_move_ms,
        rt_last_move_ms: _rt_last_move_ms,
        movement_duration_ms: movement_duration_ms,
        trial_onset_elapsed: _trial_onset_elapsed,
        trial_duration_ms: _trial_duration_ms,
        miss_reason: miss_reason,
        true_vol_param: (typeof trial.true_vol_param !== "undefined") ? trial.true_vol_param : null,
        true_stc_param: (typeof trial.true_stc_param !== "undefined") ? trial.true_stc_param : null,
        vol_level: (typeof trial.vol_level !== "undefined") ? trial.vol_level : null,
        stc_level: (typeof trial.stc_level !== "undefined") ? trial.stc_level : null
      };

      display_element.innerHTML = "";
      jsPsych.finishTrial(trial_data);
    };

    jsPsych.pluginAPI.setTimeout(function () {
      jsPsych.pluginAPI.cancelAllKeyboardResponses();

      jsPsych.pluginAPI.setTimeout(function () {
        if (!trial.is_moving_practice) {
          fly(trial);
        }
        jsPsych.pluginAPI.setTimeout(function () {
          end_trial(1);
        }, trial.drop_duration);
      }, 250); // brief locked pause before drop starts

    }, trial.response_remaining_duration);
  };

  return plugin;
})();
