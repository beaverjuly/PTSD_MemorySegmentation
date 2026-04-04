/**
 * utils.js – Shared helper functions.
 */

/** Fisher-Yates shuffle (in-place, returns array). */
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Return n random elements from arr without replacement. */
function sampleWithout(arr, n) {
  const copy = arr.slice();
  shuffle(copy);
  return copy.slice(0, Math.min(n, copy.length));
}

/** Clamp value between min and max. */
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

/**
 * Generate the reward sequence for a block.
 * @param {Object} schedule - { means, changePoints, fluctuation }
 * @param {number} numTrials
 * @returns {number[]}
 */
function generateRewardSequence(schedule, numTrials) {
  const rewards = [];
  let meanIdx = 0;

  for (let t = 0; t < numTrials; t++) {
    if (schedule.changePoints.includes(t)) {
      meanIdx++;
    }
    const mean = schedule.means[meanIdx];
    const noise = Math.round((Math.random() * 2 - 1) * schedule.fluctuation);
    rewards.push(clamp(mean + noise, 1, 99));
  }

  return rewards;
}

/**
 * Classify each trial position in the boundary block.
 * Returns labels:
 * 'far_within' | 'pre_boundary' | 'boundary' | 'post_boundary'
 */
function classifyPositions(numTrials, changePoints) {
  const labels = new Array(numTrials).fill('far_within');

  for (const cp of changePoints) {
    if (cp - 1 >= 0) labels[cp - 1] = 'pre_boundary';
    if (cp >= 0 && cp < numTrials) labels[cp] = 'boundary';
    if (cp + 1 < numTrials) labels[cp + 1] = 'post_boundary';
  }

  return labels;
}

/**
 * Sample single-item test positions for the no-boundary block.
 * Evenly spaced across the block.
 */
function sampleSingleItemNoBoundary(numTrials, count) {
  const step = numTrials / count;
  const positions = [];

  for (let i = 0; i < count; i++) {
    const pos = Math.round(i * step + step / 2);
    positions.push(clamp(pos, 0, numTrials - 1));
  }

  return shuffle(positions);
}

/**
 * Sample single-item test positions for the boundary block.
 */
function sampleSingleItemBoundary(numTrials, changePoints, quotas) {
  const labels = classifyPositions(numTrials, changePoints);
  const pools = {
    far_within: [],
    pre_boundary: [],
    boundary: [],
    post_boundary: []
  };

  labels.forEach((lbl, i) => {
    pools[lbl].push(i);
  });

  const sampled = [
    ...sampleWithout(pools.far_within, quotas.farWithin),
    ...sampleWithout(pools.pre_boundary, quotas.preBoundary),
    ...sampleWithout(pools.boundary, quotas.boundary),
    ...sampleWithout(pools.post_boundary, quotas.postBoundary)
  ];

  return shuffle(sampled);
}

/**
 * Sample paired-item test pairs for the boundary block.
 * Returns array of { posA, posB, pairType }.
 */
function samplePairedItems(changePoints, quotas) {
  const pairs = [];

  for (const cp of changePoints) {
    if (cp - 1 >= 0) {
      pairs.push({ posA: cp - 1, posB: cp, pairType: 'zero_across' });
    }
    if (cp - 1 >= 0 && cp + 1 < CONFIG.trialsPerBlock) {
      pairs.push({ posA: cp - 1, posB: cp + 1, pairType: 'one_across' });
    }
    if (cp - 2 >= 0 && cp + 2 < CONFIG.trialsPerBlock) {
      pairs.push({ posA: cp - 2, posB: cp + 2, pairType: 'three_across' });
    }
  }

  // Simple within-state defaults
  pairs.push({ posA: 1, posB: 4, pairType: 'within_state' });
  pairs.push({ posA: 2, posB: 6, pairType: 'within_state' });
  pairs.push({ posA: 10, posB: 13, pairType: 'within_state' });
  pairs.push({ posA: 19, posB: 22, pairType: 'within_state' });

  const byType = {};
  for (const p of pairs) {
    if (!byType[p.pairType]) byType[p.pairType] = [];
    byType[p.pairType].push(p);
  }

  const selected = [
    ...sampleWithout(byType.zero_across || [], quotas.zeroAcross),
    ...sampleWithout(byType.one_across || [], quotas.oneAcross),
    ...sampleWithout(byType.three_across || [], quotas.threeAcross),
    ...sampleWithout(byType.within_state || [], quotas.withinState)
  ];

  return shuffle(selected);
}

/**
 * Sample recognition items: old + new foils.
 */
function sampleRecognitionItems(encodedIds, allIds, numOld, numNew, changePoints, positionLabels) {
  const boundaryPositions = changePoints.slice();
  const boundaryIds = boundaryPositions
    .filter(p => p >= 0 && p < encodedIds.length)
    .map(p => encodedIds[p]);

  const nonBoundaryPositions = [];
  positionLabels.forEach((lbl, i) => {
    if (lbl === 'far_within') nonBoundaryPositions.push(i);
  });

  const matchedIds = sampleWithout(
    nonBoundaryPositions,
    Math.max(0, numOld - boundaryIds.length)
  ).map(p => encodedIds[p]);

  const oldItems = shuffle([...boundaryIds, ...matchedIds]).slice(0, numOld);

  const encodedSet = new Set(encodedIds);
  const foilPool = allIds.filter(id => !encodedSet.has(id));
  const newItems = sampleWithout(foilPool, numNew);

  return { oldItems, newItems };
}

/**
 * Convert trial index (0..N-1) to bar fill proportion (0..1).
 */
function trialToBarFill(trialIndex, totalTrials) {
  return (trialIndex + 1) / totalTrials;
}

/**
 * Interpolate bar color for a given proportion.
 */
function barColor(proportion) {
  const s = CONFIG.bar.colorStart;
  const e = CONFIG.bar.colorEnd;

  const r = Math.round(s[0] + (e[0] - s[0]) * proportion);
  const g = Math.round(s[1] + (e[1] - s[1]) * proportion);
  const b = Math.round(s[2] + (e[2] - s[2]) * proportion);

  return `rgb(${r},${g},${b})`;
}

/**
 * Build a gradient string for the filled portion of the bar.
 */
function barGradient(fillProportion) {
  if (fillProportion <= 0) return 'transparent';
  const startCol = barColor(0);
  const endCol = barColor(fillProportion);
  return `linear-gradient(to right, ${startCol}, ${endCol})`;
}

/**
 * Preload images.
 */
function preloadImages(filenames, dir) {
  return Promise.all(
    filenames.map(fn => {
      return new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(fn);
        img.onerror = () => resolve(fn);
        img.src = dir + fn;
      });
    })
  );
}

/**
 * Save a single trial data row.
 * In dev mode with noSave=true, do not persist into CONFIG.dataLog.
 */
function logTrial(data) {
  const row = {
    timestamp: Date.now(),
    participantId: CONFIG.participantId,
    version: CONFIG.version,
    ...data
  };

  if (CONFIG.dev.enabled && CONFIG.dev.noSave) {
    console.log('DEV NO-SAVE:', row);
    return;
  }

  CONFIG.dataLog.push(row);
}

/**
 * Download collected data as JSON.
 * In dev mode with noSave=true, skip download.
 */
function downloadData() {
  if (CONFIG.dev.enabled && CONFIG.dev.noSave) {
    console.log('DEV NO-SAVE: skipping downloadData()');
    return;
  }

  const blob = new Blob(
    [JSON.stringify(CONFIG.dataLog, null, 2)],
    { type: 'application/json' }
  );

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `pilot_${CONFIG.participantId || 'unknown'}_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Generate a simple random participant ID.
 */
function generatePID() {
  return 'P' + Math.random().toString(36).substring(2, 8).toUpperCase();
}
