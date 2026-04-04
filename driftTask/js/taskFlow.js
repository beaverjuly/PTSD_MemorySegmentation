/**
 * taskFlow.js – Experiment controller.
 * Decides what happens next; calls Render to display screens.
 */

const TaskFlow = (() => {
  let items = [];
  let instructions = {};
  let blockOrder = [];
  let currentBlockIndex = 0;

  let encodedItems = [];
  let rewardSequence = [];

  /**
   * Boot task.
   */
  async function start() {
    applyURLParams();

    if (!CONFIG.participantId) {
      CONFIG.participantId = generatePID();
    }

    const [itemData, instrData] = await Promise.all([
      fetch('data/items.json').then(r => r.json()),
      fetch('data/instructions.json').then(r => r.json())
    ]);

    items = itemData;
    instructions = instrData;

    Render.init();

    blockOrder = (CONFIG.blockOrder === 'b_first')
      ? ['boundary', 'noBoundary']
      : ['noBoundary', 'boundary'];

    const filenames = items.map(it => it.filename);
    preloadImages(filenames, CONFIG.imgDir).catch(() => {});

    if (CONFIG.dev.enabled) {
      runDevStage(CONFIG.dev.stage);
    } else {
      showWelcome();
    }
  }

  /* ===========================================================
   * NORMAL FLOW
   * =========================================================== */

  function showWelcome() {
    const versionIntro = CONFIG.version === 'rich'
      ? safeInstr('version_rich_intro', 'This is the rich version of the task.')
      : safeInstr('version_simple_intro', 'This is the simple version of the task.');

    Render.showInstruction(
      safeInstr('welcome', 'Welcome to the task.'),
      () => {
        Render.showInstruction(versionIntro, () => beginBlock(0));
      }
    );
  }

  function beginBlock(index, options = {}) {
    currentBlockIndex = index;

    const blockType = options.blockType || blockOrder[index];
    const introText = blockType === 'boundary'
      ? safeInstr('block_boundary_intro', 'You are about to begin the boundary block.')
      : safeInstr('block_no_boundary_intro', 'You are about to begin the no-boundary block.');

    prepareBlockState(blockType);

    Render.showInstruction(introText, () => runEncoding(0, { ...options, blockType }));
  }

  function prepareBlockState(blockType) {
    const pool = shuffle(items.slice());
    encodedItems = pool.slice(0, CONFIG.trialsPerBlock);

    const schedule = blockType === 'boundary' ? CONFIG.boundary : CONFIG.noBoundary;
    rewardSequence = generateRewardSequence(schedule, CONFIG.trialsPerBlock);
  }

  /* ===========================================================
   * DEV FLOW
   * =========================================================== */

  function runDevStage(stage) {
    console.log('DEV MODE:', stage);

    switch (stage) {
      case 'instructions':
        runDevInstructions();
        break;

      case 'encoding':
        runDevEncodingOnly(CONFIG.dev.blockType);
        break;

      case 'test':
        runDevMemoryOnly(CONFIG.dev.blockType);
        break;

      case 'recognition':
        runDevRecognitionOnly(CONFIG.dev.blockType);
        break;

      case 'block1':
        runDevFullBlock('noBoundary');
        break;

      case 'block2':
        runDevFullBlock('boundary');
        break;

      case 'consent':
        Render.showInstruction('DEV: Consent screen placeholder.', () => {});
        break;

      case 'screening':
        Render.showInstruction('DEV: Screening questions placeholder.', () => {});
        break;

      case 'practice':
        Render.showInstruction('DEV: Practice block placeholder.', () => {});
        break;

      case 'attention':
        Render.showInstruction('DEV: Attention-check screen placeholder.', () => {});
        break;

      default:
        Render.showInstruction('DEV MODE ACTIVE\nUnknown stage.', () => {});
        break;
    }
  }

  function runDevInstructions() {
    const msg = [
      'DEV MODE: Instructions only',
      '',
      `Version: ${CONFIG.version}`,
      `Block order: ${CONFIG.blockOrder}`,
      `Save disabled: ${CONFIG.dev.noSave ? 'yes' : 'no'}`
    ].join('\n');

    Render.showInstruction(msg, () => {});
  }

  function runDevEncodingOnly(blockType = 'noBoundary') {
    prepareBlockState(blockType);
    Render.showInstruction(
      `DEV MODE: Encoding only\nBlock: ${blockType}`,
      () => runEncoding(0, {
        blockType,
        stopAfterEncoding: true,
        devLabel: true
      })
    );
  }

  function runDevMemoryOnly(blockType = 'noBoundary') {
    prepareBlockState(blockType);
    Render.showInstruction(
      `DEV MODE: Memory tests only\nBlock: ${blockType}`,
      () => {
        if (blockType === 'boundary') {
          runBoundaryTests({ blockType, finishToDevEnd: true });
        } else {
          runNoBoundaryTests({ blockType, finishToDevEnd: true });
        }
      }
    );
  }

  function runDevRecognitionOnly(blockType = 'boundary') {
    prepareBlockState(blockType);
    Render.showInstruction(
      `DEV MODE: Recognition only\nBlock: ${blockType}`,
      () => {
        const posLabels = classifyPositions(
          CONFIG.trialsPerBlock,
          blockType === 'boundary' ? CONFIG.boundary.changePoints : []
        );
        const allIds = items.map(it => it.id);
        const encodedIds = encodedItems.map(it => it.id);

        const recog = sampleRecognitionItems(
          encodedIds,
          allIds,
          CONFIG.recognitionOld,
          CONFIG.recognitionNew,
          blockType === 'boundary' ? CONFIG.boundary.changePoints : [],
          posLabels
        );

        runRecognitionBlock(recog, () => showDevEnd());
      }
    );
  }

  function runDevFullBlock(blockType) {
    prepareBlockState(blockType);
    Render.showInstruction(
      `DEV MODE: Full block\nBlock: ${blockType}`,
      () => runEncoding(0, { blockType, finishToDevEnd: true })
    );
  }

  function showDevEnd() {
    Render.showEnd('DEV MODE complete.\nNo data file was downloaded.');
  }

  /* ===========================================================
   * ENCODING
   * =========================================================== */

function runEncoding(trialIdx, options = {}) {
  const blockType = options.blockType || blockOrder[currentBlockIndex];

  if (trialIdx >= CONFIG.trialsPerBlock) {
    if (options.stopAfterEncoding) {
      showDevEnd();
    } else {
      afterEncoding(options);
    }
    return;
  }

  const item = encodedItems[trialIdx];
  const value = rewardSequence[trialIdx];

  Render.showItemOnly(item.filename, trialIdx);

  setTimeout(() => {
    Render.showItemValue(item.filename, value, trialIdx);

    setTimeout(() => {
      logTrial({
        phase: 'encoding',
        block: blockType,
        trialIndex: trialIdx,
        itemId: item.id,
        filename: item.filename,
        rewardValue: value,
        barFill: trialToBarFill(trialIdx, CONFIG.trialsPerBlock),
        isDev: CONFIG.dev.enabled
      });

      Render.showEncodingITI(trialIdx);

      setTimeout(() => {
        runEncoding(trialIdx + 1, options);
      }, CONFIG.iti);

    }, CONFIG.itemValueDuration);
  }, CONFIG.itemOnlyDuration);
}

  function afterEncoding(options = {}) {
    const blockType = options.blockType || blockOrder[currentBlockIndex];

    if (blockType === 'noBoundary') {
      runNoBoundaryTests(options);
    } else {
      runBoundaryTests(options);
    }
  }

  /* ===========================================================
   * NO-BOUNDARY TEST BATTERY
   * =========================================================== */

  function runNoBoundaryTests(options = {}) {
    Render.showInstruction(
      safeInstr('single_item_intro', 'You will now do single-item placement.'),
      () => {
        const testPositions = sampleSingleItemNoBoundary(
          CONFIG.trialsPerBlock,
          CONFIG.singleItemCount
        );

        runSinglePlacementBlock(testPositions, options, () => {
          Render.showInstruction(
            safeInstr('paired_item_intro', 'You will now do paired-item placement.'),
            () => {
              const pairs = createNoBoundaryPairs();
              runPairedPlacementBlock(pairs, options, () => {
                finishBlock(options);
              });
            }
          );
        });
      }
    );
  }

  function createNoBoundaryPairs() {
    const pairs = [];
    const step = Math.floor(CONFIG.trialsPerBlock / 8);

    for (let i = 0; i < 8; i++) {
      const posA = i * step;
      const posB = Math.min(i * step + Math.floor(step / 2), CONFIG.trialsPerBlock - 1);
      pairs.push({
        posA,
        posB,
        pairType: 'within_block'
      });
    }

    return shuffle(pairs);
  }

  /* ===========================================================
   * BOUNDARY TEST BATTERY
   * =========================================================== */

  function runBoundaryTests(options = {}) {
    Render.showInstruction(
      safeInstr('recognition_intro', 'You will now do recognition.'),
      () => {
        const posLabels = classifyPositions(CONFIG.trialsPerBlock, CONFIG.boundary.changePoints);
        const allIds = items.map(it => it.id);
        const encodedIds = encodedItems.map(it => it.id);

        const recog = sampleRecognitionItems(
          encodedIds,
          allIds,
          CONFIG.recognitionOld,
          CONFIG.recognitionNew,
          CONFIG.boundary.changePoints,
          posLabels
        );

        runRecognitionBlock(recog, () => {
          Render.showInstruction(
            safeInstr('single_item_intro', 'You will now do single-item placement.'),
            () => {
              const testPositions = sampleSingleItemBoundary(
                CONFIG.trialsPerBlock,
                CONFIG.boundary.changePoints,
                CONFIG.singleItemBoundary
              );

              runSinglePlacementBlock(testPositions, options, () => {
                Render.showInstruction(
                  safeInstr('paired_item_intro', 'You will now do paired-item placement.'),
                  () => {
                    const pairs = samplePairedItems(
                      CONFIG.boundary.changePoints,
                      CONFIG.pairedItemBoundary
                    );

                    runPairedPlacementBlock(pairs, options, () => {
                      finishBlock(options);
                    });
                  }
                );
              });
            }
          );
        });
      }
    );
  }

  /* ===========================================================
   * TEST RUNNERS
   * =========================================================== */

  function runSinglePlacementBlock(positions, options, onDone) {
    let idx = 0;
    const blockType = options.blockType || blockOrder[currentBlockIndex];

    function next() {
      if (idx >= positions.length) {
        onDone();
        return;
      }

      const pos = positions[idx];
      const item = encodedItems[pos];
      const startTime = Date.now();

      Render.showSinglePlacement(item.filename, CONFIG.trialsPerBlock, (proportion) => {
        const labels = classifyPositions(
          CONFIG.trialsPerBlock,
          blockType === 'boundary' ? CONFIG.boundary.changePoints : []
        );

        logTrial({
          phase: 'single_placement',
          block: blockType,
          trialIndex: pos,
          itemId: item.id,
          filename: item.filename,
          truePosition: pos / CONFIG.trialsPerBlock,
          response: proportion,
          error: Math.abs(proportion - pos / CONFIG.trialsPerBlock),
          rt: Date.now() - startTime,
          positionLabel: labels[pos],
          isDev: CONFIG.dev.enabled
        });

        idx++;
        setTimeout(next, CONFIG.iti);
      });
    }

    next();
  }

  function runPairedPlacementBlock(pairs, options, onDone) {
    let idx = 0;
    const blockType = options.blockType || blockOrder[currentBlockIndex];

    function next() {
      if (idx >= pairs.length) {
        onDone();
        return;
      }

      const pair = pairs[idx];
      const itemA = encodedItems[pair.posA];
      const itemB = encodedItems[pair.posB];
      const startTime = Date.now();

      Render.showPairedPlacement(itemA.filename, itemB.filename, CONFIG.trialsPerBlock, (resp) => {
        const truePosA = pair.posA / CONFIG.trialsPerBlock;
        const truePosB = pair.posB / CONFIG.trialsPerBlock;
        const trueOrder = pair.posA < pair.posB ? 'A_first' : 'B_first';
        const respOrder = resp.posA < resp.posB ? 'A_first' : 'B_first';

        logTrial({
          phase: 'paired_placement',
          block: blockType,
          pairType: pair.pairType,
          posA: pair.posA,
          posB: pair.posB,
          itemIdA: itemA.id,
          itemIdB: itemB.id,
          truePosA,
          truePosB,
          trueDistance: Math.abs(truePosB - truePosA),
          responsePosA: resp.posA,
          responsePosB: resp.posB,
          responseDistance: Math.abs(resp.posB - resp.posA),
          orderCorrect: trueOrder === respOrder,
          rt: Date.now() - startTime,
          isDev: CONFIG.dev.enabled
        });

        idx++;
        setTimeout(next, CONFIG.iti);
      });
    }

    next();
  }

  function runRecognitionBlock(recog, onDone) {
    const trials = shuffle([
      ...recog.oldItems.map(id => ({ itemId: id, isOld: true })),
      ...recog.newItems.map(id => ({ itemId: id, isOld: false }))
    ]);

    let idx = 0;
    const blockType = blockOrder[currentBlockIndex] || CONFIG.dev.blockType || 'boundary';

    function next() {
      if (idx >= trials.length) {
        onDone();
        return;
      }

      const trial = trials[idx];
      const item = items.find(it => it.id === trial.itemId);
      const startTime = Date.now();

      Render.showRecognition(item.filename, (resp) => {
        const correct = (resp === 'old') === trial.isOld;

        logTrial({
          phase: 'recognition',
          block: blockType,
          itemId: trial.itemId,
          filename: item.filename,
          isOld: trial.isOld,
          response: resp,
          correct,
          rt: Date.now() - startTime,
          isDev: CONFIG.dev.enabled
        });

        idx++;
        setTimeout(next, CONFIG.iti);
      });
    }

    next();
  }

  /* ===========================================================
   * BLOCK TRANSITIONS
   * =========================================================== */

  function finishBlock(options = {}) {
    if (CONFIG.dev.enabled && options.finishToDevEnd) {
      showDevEnd();
      return;
    }

    if (currentBlockIndex < blockOrder.length - 1) {
      Render.showInstruction(
        safeInstr('break_text', 'Take a short break.'),
        () => beginBlock(currentBlockIndex + 1)
      );
    } else {
      endExperiment();
    }
  }

  function endExperiment() {
    downloadData();
    Render.showEnd(safeInstr('end_text', 'Thank you for participating.'));
  }

  /* ===========================================================
   * HELPERS
   * =========================================================== */

  function safeInstr(key, fallback) {
    return (instructions && instructions[key]) ? instructions[key] : fallback;
  }

  return { start };
})();

/* ---- Boot ---- */
document.addEventListener('DOMContentLoaded', () => {
  TaskFlow.start();
});
