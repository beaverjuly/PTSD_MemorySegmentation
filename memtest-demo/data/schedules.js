// Default reward schedule data
const BLOCK_LENGTH = 27;

const SCHEDULE_VERSIONS = {
  A: { label: 'Version A (9/9/9)', boundaries: [10, 19], stateLengths: [9, 9, 9] },
  B: { label: 'Version B (8/10/9)', boundaries: [9, 19], stateLengths: [8, 10, 9] },
  C: { label: 'Version C (10/8/9)', boundaries: [11, 19], stateLengths: [10, 8, 9] },
};

const DEFAULT_STATE_MEANS = [30, 60, 40];
const DEFAULT_BOUNDARIES = [10, 19];
const REWARD_NOISE_RANGE = 5; // +/- from mean

function generateRewards(means, boundaries) {
  const rewards = [];
  for (let i = 1; i <= BLOCK_LENGTH; i++) {
    let stateIdx;
    if (i < boundaries[0]) stateIdx = 0;
    else if (i < boundaries[1]) stateIdx = 1;
    else stateIdx = 2;
    const mean = means[stateIdx];
    const noise = Math.round((Math.random() - 0.5) * 2 * REWARD_NOISE_RANGE);
    rewards.push(mean + noise);
  }
  return rewards;
}

function getStateForPosition(pos, boundaries) {
  if (pos < boundaries[0]) return 0;
  if (pos < boundaries[1]) return 1;
  return 2;
}
