/**
 * config.js – Single source of truth for task parameters.
 * Edit values here; do not hardcode them in taskFlow or render.
 */

const CONFIG = {
  /* ---- Version ---- */
  // Set programmatically or via URL param ?version=simple|rich
  version: 'simple', // 'simple' | 'rich'

  /* ---- Dev mode ---- */
  dev: {
    enabled: false,
    stage: null,
    noSave: true,
    useMockData: true,
    blockType: 'noBoundary' // 'noBoundary' | 'boundary'
  },

  /* ---- Block structure ---- */
  trialsPerBlock: 36,
  numStates: 4, // boundary block only
  trialsPerState: 9,

  /* ---- Timing (ms) ---- */
  itemOnlyDuration: 1000,  // Phase 1: item alone
  itemValueDuration: 2000, // Phase 2: item + value
  iti: 500,                // inter-trial interval (blank)
  fixationDuration: 500,

  /* ---- Reward schedule ---- */
  noBoundary: {
    means: [50],
    changePoints: [],
    fluctuation: 5
  },
  boundary: {
    means: [30, 70, 20, 80],
    changePoints: [9, 18, 27], // first trial of each new state
    fluctuation: 5
  },

  /* ---- Blue bar (Version B / rich) ---- */
  bar: {
    show: false, // toggled by version selection
    width: '80%',
    height: 28,
    borderColor: '#b0b8c4',
    emptyColor: '#e8ecf0',
    colorStart: [200, 220, 245],
    colorEnd: [30, 70, 140],
    bottomOffset: 40
  },

  /* ---- Image settings ---- */
  imgDir: 'assets/img/',
  imgSize: 220,
  placeholderColor: '#d0d0d0',

  /* ---- Test sampling ---- */
  singleItemCount: 10,
  pairedItemCount: 8,
  recognitionOld: 12,
  recognitionNew: 12,

  /* ---- Boundary-block single-item sample ---- */
  singleItemBoundary: {
    farWithin: 4,
    preBoundary: 2,
    boundary: 2,
    postBoundary: 2
  },

  /* ---- Paired-item boundary pairs ---- */
  pairedItemBoundary: {
    withinState: 2,
    zeroAcross: 2,
    oneAcross: 2,
    threeAcross: 2
  },

  /* ---- Response scale ---- */
  scale: {
    width: '80%',
    height: 36,
    plainColor: '#999',
    tickCount: 0
  },

  /* ---- Keys ---- */
  keys: {
    next: 'Enter',
    oldKey: 'f',
    newKey: 'j',
    oldLabel: 'F = OLD',
    newLabel: 'J = NEW'
  },

  /* ---- Counterbalancing ---- */
  // 'nb_first' | 'b_first'
  blockOrder: 'nb_first',

  /* ---- Data output ---- */
  participantId: null,
  dataLog: []
};

/**
 * Apply version settings.
 */
function applyVersion(ver) {
  const safeVer = (ver === 'rich') ? 'rich' : 'simple';
  CONFIG.version = safeVer;
  CONFIG.bar.show = (safeVer === 'rich');
}

/**
 * Read URL parameters for runtime configuration.
 *
 * Supported params:
 * ?version=simple|rich
 * ?order=nb_first|b_first
 * ?pid=ABC123
 * ?dev=1
 * ?stage=instructions|encoding|test|block1|block2|recognition|consent|screening|practice|attention
 * ?block=noBoundary|boundary
 * ?save=1   // only relevant in dev; default is no save
 */
function applyURLParams() {
  const params = new URLSearchParams(window.location.search);

  if (params.has('version')) {
    applyVersion(params.get('version'));
  } else {
    applyVersion(CONFIG.version);
  }

  if (params.has('order')) {
    const val = params.get('order');
    CONFIG.blockOrder = (val === 'b_first') ? 'b_first' : 'nb_first';
  }

  if (params.has('pid')) {
    CONFIG.participantId = params.get('pid');
  }

  if (params.get('dev') === '1') {
    CONFIG.dev.enabled = true;
    CONFIG.dev.stage = params.get('stage') || 'instructions';
    CONFIG.dev.noSave = params.get('save') === '1' ? false : true;

    const blockParam = params.get('block');
    if (blockParam === 'boundary' || blockParam === 'noBoundary') {
      CONFIG.dev.blockType = blockParam;
    }

    // Helpful aliases
    if (CONFIG.dev.stage === 'block1') CONFIG.dev.blockType = 'noBoundary';
    if (CONFIG.dev.stage === 'block2') CONFIG.dev.blockType = 'boundary';
  }
}
