// Token metadata and default recognition streams

const TOKEN_CATEGORIES = {
  critical: { color: '#e74c3c', label: 'Critical Pair' },
  within: { color: '#4caf50', label: 'Within-State Filler' },
  single: { color: '#aaaaaa', label: 'Unused Old' },
  new_item: { color: '#d5d5d5', label: 'New' },
};

const TOKEN_DEFS = {
  // Block 1 critical tokens
  'B1→P1': {
    label: 'B1→P1',
    expanded: 'Boundary 1 → Predecessor 1',
    items: [10, 9],
    category: 'critical',
  },
  'B1→S1': {
    label: 'B1→S1',
    expanded: 'Boundary 1 → Successor 1',
    items: [10, 11],
    category: 'critical',
  },
  'B2→P2': {
    label: 'B2→P2',
    expanded: 'Boundary 2 → Predecessor 2',
    items: [19, 18],
    category: 'critical',
  },
  'B2→S2': {
    label: 'B2→S2',
    expanded: 'Boundary 2 → Successor 2',
    items: [19, 20],
    category: 'critical',
  },

  // Block 2 critical tokens
  'P1→B1': {
    label: 'P1→B1',
    expanded: 'Predecessor 1 → Boundary 1',
    items: [9, 10],
    category: 'critical',
  },
  'P1→S1': {
    label: 'P1→S1',
    expanded: 'Predecessor 1 → Successor 1 (across boundary)',
    items: [9, 11],
    category: 'critical',
  },
  'P2→B2': {
    label: 'P2→B2',
    expanded: 'Predecessor 2 → Boundary 2',
    items: [18, 19],
    category: 'critical',
  },
  'P2→S2': {
    label: 'P2→S2',
    expanded: 'Predecessor 2 → Successor 2 (across boundary)',
    items: [18, 20],
    category: 'critical',
  },

  // Within-state fillers
  'W1': {
    label: 'W1',
    expanded: 'Within-State Filler Pair 1',
    items: [4, 5],
    category: 'within',
  },
  'W2': {
    label: 'W2',
    expanded: 'Within-State Filler Pair 2',
    items: [13, 14],
    category: 'within',
  },

  // Single old items from otherwise unused positions
  'O1': {
    label: 'O1',
    expanded: 'Unused Old Item 1',
    items: [2],
    category: 'single',
  },
  'O2': {
    label: 'O2',
    expanded: 'Unused Old Item 2',
    items: [26],
    category: 'single',
  },

  // New items
  'New': {
    label: 'New',
    expanded: 'New Item',
    items: [],
    category: 'new_item',
  },
};

// Default Block 1 stream
const DEFAULT_BLOCK1_STREAM = [
  'New',    // N1
  'B1→P1',
  'New',    // N2
  'W1',
  'New',    // N3
  'O1',
  'New',    // N4
  'B2→S2',
  'New',    // N5
  'W2',
  'New',    // N6
  'O2',
  'New',    // N7
  'B1→S1',
  'New',    // N8
  'New',    // N9
  'New',    // N10
  'B2→P2',
];

// Default Block 2 stream
const DEFAULT_BLOCK2_STREAM = [
  'New',    // N1
  'P1→B1',
  'New',    // N2
  'W1',
  'New',    // N3
  'O1',
  'New',    // N4
  'P2→S2',
  'New',    // N5
  'W2',
  'New',    // N6
  'O2',
  'New',    // N7
  'P1→S1',
  'New',    // N8
  'New',    // N9
  'New',    // N10
  'P2→B2',
];

// Tokens available for each block
const BLOCK1_TOKENS = ['B1→P1', 'B1→S1', 'B2→P2', 'B2→S2', 'W1', 'W2', 'O1', 'O2', 'New'];
const BLOCK2_TOKENS = ['P1→B1', 'P1→S1', 'P2→B2', 'P2→S2', 'W1', 'W2', 'O1', 'O2', 'New'];