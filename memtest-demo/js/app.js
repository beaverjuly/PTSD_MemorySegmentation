// Main state controller

const App = (() => {
  let currentSection = 'preview';
  let currentPreview = 'encoding-a';

  function init() {
    // Tab navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        currentSection = tab.dataset.section;
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById('section-' + currentSection).classList.add('active');
      });
    });

    // Preview sub-navigation
    document.querySelectorAll('.preview-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        currentPreview = tab.dataset.preview;
        document.querySelectorAll('.preview-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        renderPreview();
      });
    });

    // Init modules
    Preview.init(document.getElementById('preview-area'));
    ScheduleEditor.init(document.getElementById('schedule-editor'));
    StreamBuilder.init(document.getElementById('stream-builder'));

    renderPreview();
  }

  function renderPreview() {
    switch (currentPreview) {
      case 'encoding-a': Preview.renderEncodingScreen('A'); break;
      case 'encoding-b': Preview.renderEncodingScreen('B'); break;
      case 'block1-recog': Preview.renderRecognitionScreen(1); break;
      case 'block2-recog': Preview.renderRecognitionScreen(2); break;
      case 'block3': Preview.renderBlock3(true); break;
    }
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
