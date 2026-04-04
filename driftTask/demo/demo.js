/**
 * demo.js – Demo state controller and preview renderer.
 * Renders static representative snapshots; no real task timing.
 */

(function () {
  /* ---- Demo state ---- */
  const state = {
    version: 'simple',
    block: 'noBoundary',
    screen: 'encoding',
    phase: 'itemOnly'
  };

  /* Representative mock data */
  const mockFilename = 'backpack.jpg';        // placeholder item
  const mockFilename2 = 'camera01b.jpg';      // second item for pairs
  const mockValue = 47;
  const mockTrialIndex = 14;                  // mid-block position

  /* ---- Initialise ---- */
document.addEventListener('DOMContentLoaded', () => {
  CONFIG.imgDir = '../assets/img/';  
  Render.init();
  setupControls();
  renderPreview();
});

  /* ---- Button wiring ---- */
  function setupControls() {
    document.querySelectorAll('.btn-group').forEach(group => {
      const key = group.dataset.group;
      group.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
          group.querySelectorAll('button').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state[key] = btn.dataset.val;

          // Apply version
          if (key === 'version') applyVersion(state.version);

          renderPreview();
        });
      });
    });
  }

  /* ---- Main preview dispatcher ---- */
  function renderPreview() {
    // Ensure config matches state
    applyVersion(state.version);

    switch (state.screen) {
      case 'encoding':
        renderEncoding();
        break;
      case 'single':
        renderSinglePlacement();
        break;
      case 'paired':
        renderPairedPlacement();
        break;
      case 'recognition':
        renderRecognition();
        break;
    }
  }

  function renderEncoding() {
    if (state.phase === 'itemOnly') {
      Render.showItemOnly(mockFilename, mockTrialIndex);
    } else {
      Render.showItemValue(mockFilename, mockValue, mockTrialIndex);
    }
  }

  function renderSinglePlacement() {
    Render.showSinglePlacement(mockFilename, CONFIG.trialsPerBlock, (prop) => {
      console.log('Demo single placement:', prop);
    });
  }

  function renderPairedPlacement() {
    Render.showPairedPlacement(mockFilename, mockFilename2, CONFIG.trialsPerBlock, (resp) => {
      console.log('Demo paired placement:', resp);
    });
  }

  function renderRecognition() {
    Render.showRecognition(mockFilename, (resp) => {
      console.log('Demo recognition:', resp);
    });
  }
})();
