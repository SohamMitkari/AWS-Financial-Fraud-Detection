/**
 * Tests for Real Dataset Mode (Stages 1-5)
 * Covers:
 *  - Streaming CSV reader: chunk boundaries, quotes, escaped quotes, blanks -> NaN
 *  - 300,000 row cap: stops reader, sets capped = true, exact row count
 *  - Auto-detect: label & order column detection, 0/1 fallback, none option, dropped label rows, non-numeric exclusion
 *  - Partition splits: chronological stable sort, 70/30/20 split proportions, seeded mulberry32(42) determinism
 *  - Fixture testing across all 5 sample fixtures (tests/fixtures/real/*_sample300.csv)
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

// 1. Load real mode functions from frontend/index.html following repo extraction conventions
function loadRealFunctions() {
  const htmlPath = path.resolve(__dirname, '../frontend/index.html');
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');
  const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>[\s\S]*?$/);
  if (!scriptMatch) throw new Error('Failed to find <script> in frontend/index.html');
  const scriptCode = scriptMatch[1];

  function extractFunction(code, fnName) {
    const re = new RegExp('\\b(?:async\\s+)?function\\s+' + fnName + '\\s*\\(', 'g');
    const match = re.exec(code);
    if (!match) return null;
    const startPos = match.index;
    const parenStart = code.indexOf('(', startPos);
    let pDepth = 0;
    let braceStart = -1;
    for (let i = parenStart; i < code.length; i++) {
      if (code[i] === '(') pDepth++;
      else if (code[i] === ')') {
        pDepth--;
        if (pDepth === 0) {
          braceStart = code.indexOf('{', i);
          break;
        }
      }
    }
    if (braceStart === -1) return null;

    let depth = 0;
    let inString = false;
    let stringChar = '';
    let inLineComment = false;
    let inBlockComment = false;

    for (let i = braceStart; i < code.length; i++) {
      const ch = code[i];
      const next = code[i + 1];

      if (inLineComment) {
        if (ch === '\n') inLineComment = false;
        continue;
      }
      if (inBlockComment) {
        if (ch === '*' && next === '/') {
          inBlockComment = false;
          i++;
        }
        continue;
      }
      if (inString) {
        if (ch === '\\') {
          i++;
        } else if (ch === stringChar) {
          inString = false;
        }
        continue;
      }

      if (ch === '/' && next === '/') {
        inLineComment = true;
        i++;
        continue;
      }
      if (ch === '/' && next === '*') {
        inBlockComment = true;
        i++;
        continue;
      }

      if (ch === '"' || ch === "'" || ch === '`') {
        inString = true;
        stringChar = ch;
        continue;
      }

      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) return code.slice(startPos, i + 1);
      }
    }
    return null;
  }

  const fnNames = [
    'mulberry32',
    'streamParseCSV',
    'autoDetectColumns',
    'processRealColumns',
    'cleanRealDataset',
    'computeRealEDA',
    'splitRealDataset',
    'computePrAucJS',
    'computeRocAucJS',
    'flattenTree',
    'prepareModelTrees',
    'scoreRow',
    'sigmoidJS',
    'buildRealTrainPayload',
    'computeGainCurve',
    'computeRealReplay',
    'simulateRealStreamReplay'
  ];

  const snippets = [];
  for (const name of fnNames) {
    const fnCode = extractFunction(scriptCode, name);
    if (!fnCode) throw new Error(`Could not extract ${name} from frontend/index.html`);
    snippets.push(fnCode);
  }

  const sandbox = {
    console,
    Math,
    Number,
    String,
    Array,
    Map,
    Set,
    Uint8Array,
    Int16Array,
    Int32Array,
    Float64Array,
    TextDecoder,
    TextEncoder,
    JSON,
    isNaN,
    Infinity,
    state: {}
  };

  const context = vm.createContext(sandbox);
  vm.runInContext(snippets.join('\n\n'), context);
  return sandbox;
}

const {
  mulberry32,
  streamParseCSV,
  autoDetectColumns,
  processRealColumns,
  cleanRealDataset,
  computeRealEDA,
  splitRealDataset,
  computePrAucJS,
  computeRocAucJS,
  flattenTree,
  prepareModelTrees,
  scoreRow,
  sigmoidJS,
  buildRealTrainPayload,
  computeGainCurve,
  computeRealReplay,
  simulateRealStreamReplay,
  state
} = loadRealFunctions();

// Helper for cross-realm deep comparisons in VM
function assertDeep(actual, expected, msg) {
  assert.deepStrictEqual(JSON.parse(JSON.stringify(actual)), JSON.parse(JSON.stringify(expected)), msg);
}

// Helper to wrap buffer/string into a web-compatible ReadableStream
function makeStream(data, chunkSize = 64) {
  const encoder = new TextEncoder();
  const bytes = typeof data === 'string' ? encoder.encode(data) : data;
  let offset = 0;
  let cancelled = false;

  return {
    getReader() {
      return {
        async read() {
          if (cancelled || offset >= bytes.length) {
            return { done: true, value: undefined };
          }
          const end = Math.min(bytes.length, offset + chunkSize);
          const chunk = bytes.subarray(offset, end);
          offset = end;
          return { done: false, value: chunk };
        },
        async cancel() {
          cancelled = true;
        }
      };
    }
  };
}

async function runTests() {
  console.log('=== REAL DATASET MODE JS UNIT TESTS ===\n');

  // TEST SUITE 1: Streaming CSV Reader Chunk Boundaries & Escaping
  console.log('1. Testing streamParseCSV chunk boundaries and edge cases...');
  {
    // CSV with quotes, commas in quotes, escaped quotes, blank cells, newlines across chunk boundaries
    const rawCsv = [
      'order,amount,name,label',
      '1,100.5,"Acme, Inc.",0',
      '2,,"Hello ""World""",1',
      '3,250.75,"Simple",0',
      '4,0.0,,1',
      '5,-15.5,"End, quote test",0\n'
    ].join('\r\n');

    // Test with tiny chunk sizes (1 byte, 2 bytes, 7 bytes, 13 bytes, 1024 bytes)
    for (const chunkSize of [1, 2, 7, 13, 1024]) {
      const stream = makeStream(rawCsv, chunkSize);
      const res = await streamParseCSV(stream);

      assert.strictEqual(res.rawRowCount, 5, `Expected 5 rows for chunkSize ${chunkSize}`);
      assertDeep(res.headers, ['order', 'amount', 'name', 'label']);

      const amountArr = res.rawColumns.get('amount');
      const labelArr = res.rawColumns.get('label');
      const orderArr = res.rawColumns.get('order');

      assert.strictEqual(orderArr[0], 1);
      assert.strictEqual(amountArr[0], 100.5);
      assert.strictEqual(labelArr[0], 0);

      // Blank amount at row index 1 -> NaN
      assert.strictEqual(isNaN(amountArr[1]), true, 'Blank cell must parse to NaN');
      assert.strictEqual(res.blankMap.get('amount'), 1, 'Blank map count must be 1');

      // Sample preview string preservation
      assert.strictEqual(res.sampleRows[0].name, 'Acme, Inc.');
      assert.strictEqual(res.sampleRows[1].name, 'Hello "World"');
      assert.strictEqual(res.sampleRows[3].name, '');
    }
    console.log('   ✓ Chunk boundary, quoted field, escaped quote, and blank handling passed.');
  }

  // TEST SUITE 2: 300,000 Row Cap
  console.log('2. Testing 300,000 row cap enforcement...');
  {
    // Generate a simulated stream with 300,050 rows
    const header = 'order,feature1,label\n';
    const rowTemplate = '1,123.45,0\n';
    const encoder = new TextEncoder();
    const headerBytes = encoder.encode(header);
    const rowBytes = encoder.encode(rowTemplate);

    let rowsServed = 0;
    const totalToServe = 300050;
    let cancelCalled = false;

    const hugeStream = {
      getReader() {
        let sentHeader = false;
        return {
          async read() {
            if (!sentHeader) {
              sentHeader = true;
              return { done: false, value: headerBytes };
            }
            if (rowsServed >= totalToServe) {
              return { done: true, value: undefined };
            }
            // Deliver batch of 5,000 rows
            const batchSize = Math.min(5000, totalToServe - rowsServed);
            const batchBuf = new Uint8Array(rowBytes.length * batchSize);
            for (let b = 0; b < batchSize; b++) {
              batchBuf.set(rowBytes, b * rowBytes.length);
            }
            rowsServed += batchSize;
            return { done: false, value: batchBuf };
          },
          async cancel() {
            cancelCalled = true;
          }
        };
      }
    };

    const cappedResult = await streamParseCSV(hugeStream, { maxRows: 300000 });
    assert.strictEqual(cappedResult.rawRowCount, 300000, 'Row count must be capped at exactly 300,000');
    assert.strictEqual(cappedResult.capped, true, 'capped flag must be true');
    assert.strictEqual(cancelCalled, true, 'reader.cancel() must be called upon hitting cap');
    console.log(`   ✓ 300,000 row cap verified: stopped at exactly ${cappedResult.rawRowCount.toLocaleString()} rows with reader.cancel().`);
  }

  // TEST SUITE 3: Auto-detect, Dropped Labels & Non-Numeric Column Exclusion
  console.log('3. Testing autoDetectColumns and processRealColumns on test fixtures...');
  {
    const fixturesDir = path.resolve(__dirname, 'fixtures/real');
    const fixtureConfigs = [
      { file: 'creditcard_sample300.csv', expLabel: 'label', expOrder: 'order', hasOrder: true },
      { file: 'elliptic_sample300.csv', expLabel: 'label', expOrder: 'order', hasOrder: true },
      { file: 'bitcoinheist_sample300.csv', expLabel: 'label', expOrder: 'order', hasOrder: true },
      { file: 'ethereum_sample300.csv', expLabel: 'label', expOrder: '__none__', hasOrder: false },
      { file: 'taiwan_sample300.csv', expLabel: 'label', expOrder: '__none__', hasOrder: false }
    ];

    for (const cfg of fixtureConfigs) {
      const filePath = path.join(fixturesDir, cfg.file);
      assert.strictEqual(fs.existsSync(filePath), true, `Fixture not found: ${cfg.file}`);
      const content = fs.readFileSync(filePath, 'utf8');
      const stream = makeStream(content, 4096);
      const parsed = await streamParseCSV(stream);

      assert.strictEqual(parsed.rawRowCount, 300, `${cfg.file} must have 300 rows`);

      const detected = autoDetectColumns(parsed.headers, parsed.rawColumns, parsed.rawRowCount, parsed.nonNumericMap);
      assert.strictEqual(detected.labelCol, cfg.expLabel, `Expected label col '${cfg.expLabel}' for ${cfg.file}, got '${detected.labelCol}'`);
      assert.strictEqual(detected.orderCol, cfg.expOrder, `Expected order col '${cfg.expOrder}' for ${cfg.file}, got '${detected.orderCol}'`);

      const processed = processRealColumns(parsed, detected.labelCol, detected.orderCol);
      assert.strictEqual(processed.rowCount, 300, 'Expected 300 valid rows');
      assert.strictEqual(processed.droppedRows, 0, 'No dropped rows expected in clean fixture');
      assert.strictEqual(processed.fraudCount > 0, true, 'Fraud count must be > 0');

      // Stage 2 Cleaning
      const cleaned = cleanRealDataset(processed);
      assert.strictEqual(cleaned.rowCount, 300);
      assert.strictEqual(cleaned.missingStats.length <= 10, true, 'Top 10 missing % list capped at 10');

      // Stage 3 EDA
      const eda = computeRealEDA(processed, cleaned.featureCols);
      assert.strictEqual(eda.count0 + eda.count1, 300);
      assert.strictEqual(eda.count1, processed.fraudCount);
      assert.strictEqual(eda.top10.length <= 10, true);

      console.log(`   ✓ ${cfg.file}: label='${detected.labelCol}', order='${detected.orderCol}', features=${cleaned.featureCols.length}, fraud=${processed.fraudCount}/${processed.rowCount}`);
    }

    // Test alternate column name detection (class, fraud, isfraud, step, time) & 0/1 fallback
    const altCsv = 'step,V1,V2,Class\n1,0.5,1.2,0\n2,-0.5,2.1,1\n3,0.0,0.8,0\n';
    const altParsed = await streamParseCSV(makeStream(altCsv));
    const altDet = autoDetectColumns(altParsed.headers, altParsed.rawColumns, altParsed.rawRowCount, altParsed.nonNumericMap);
    assert.strictEqual(altDet.labelCol, 'Class');
    assert.strictEqual(altDet.orderCol, 'step');

    // Test fallback to 0/1 column when header is unknown
    const unkCsv = 'unknown_order,V1,V2,my_custom_binary_target\n10,0.5,1.2,0\n20,-0.5,2.1,1\n30,0.0,0.8,0\n';
    const unkParsed = await streamParseCSV(makeStream(unkCsv));
    const unkDet = autoDetectColumns(unkParsed.headers, unkParsed.rawColumns, unkParsed.rawRowCount, unkParsed.nonNumericMap);
    assert.strictEqual(unkDet.labelCol, 'my_custom_binary_target', 'Should detect single 0/1 column as label');

    // Test dropping rows with missing or non-0/1 label
    const badLabelCsv = 'order,amount,label\n1,10.0,0\n2,20.0,1\n3,30.0,\n4,40.0,2\n5,50.0,-1\n6,60.0,1\n';
    const badParsed = await streamParseCSV(makeStream(badLabelCsv));
    const badProc = processRealColumns(badParsed, 'label', 'order');
    assert.strictEqual(badProc.rowCount, 3, 'Only 3 rows have label 0 or 1');
    assert.strictEqual(badProc.droppedRows, 3, '3 rows with missing/invalid label must be dropped');

    // Test non-numeric feature column exclusion
    const textColCsv = 'order,text_id,amount,label\n1,ABC,100,0\n2,DEF,200,1\n';
    const textParsed = await streamParseCSV(makeStream(textColCsv));
    const textProc = processRealColumns(textParsed, 'label', 'order');
    assertDeep(textProc.nonNumericCols, ['text_id'], 'text_id must be excluded');
    assertDeep(textProc.featureCols, ['amount'], 'only numeric amount retained');
    console.log('   ✓ Auto-detection edge cases, dropped label rows, and non-numeric exclusions verified.');
  }

  // TEST SUITE 4: Partition Split (Chronological, 70/30/20 Proportions & Seeded Determinism)
  console.log('4. Testing Stage 5 split engine...');
  {
    const N = 300;

    // A. Chronological order column test
    const orderValues = new Float64Array(N);
    for (let i = 0; i < N; i++) orderValues[i] = (N - i) * 10; // Reverse order to test sorting

    const chronoSplit = splitRealDataset(N, orderValues, 42);
    assert.strictEqual(chronoSplit.splitMode, 'chronological');
    assert.strictEqual(chronoSplit.train.length, 210, 'Train must be 70% of 300 = 210');
    assert.strictEqual(chronoSplit.test.length, 90, 'Test must be 30% of 300 = 90');
    assert.strictEqual(chronoSplit.val.length, 42, 'Val must be 20% of 210 = 42');
    assert.strictEqual(chronoSplit.fit.length, 168, 'Fit must be 80% of 210 = 168');
    assert.strictEqual(chronoSplit.fit.length + chronoSplit.val.length, chronoSplit.train.length);

    // Verify chronological ordering:
    // First fit slice, then val slice, then test partition must have monotonically increasing order values
    const maxFitOrder = Math.max(...Array.from(chronoSplit.fit).map(idx => orderValues[idx]));
    const minValOrder = Math.min(...Array.from(chronoSplit.val).map(idx => orderValues[idx]));
    const maxValOrder = Math.max(...Array.from(chronoSplit.val).map(idx => orderValues[idx]));
    const minTestOrder = Math.min(...Array.from(chronoSplit.test).map(idx => orderValues[idx]));

    assert.strictEqual(maxFitOrder <= minValOrder, true, 'Fit order values must precede validation');
    assert.strictEqual(maxValOrder <= minTestOrder, true, 'Validation order values must precede test partition');

    // B. No order column: seeded random split (mulberry32, seed 42)
    const randomSplit1 = splitRealDataset(N, null, 42);
    const randomSplit2 = splitRealDataset(N, null, 42);

    assert.strictEqual(randomSplit1.splitMode, 'seeded');
    assert.strictEqual(randomSplit1.train.length, 210);
    assert.strictEqual(randomSplit1.test.length, 90);
    assert.strictEqual(randomSplit1.val.length, 42);
    assert.strictEqual(randomSplit1.fit.length, 168);

    // Determinism check: seed 42 must yield identical index arrays
    assertDeep(Array.from(randomSplit1.fit), Array.from(randomSplit2.fit), 'Fit indices must be identical for seed 42');
    assertDeep(Array.from(randomSplit1.val), Array.from(randomSplit2.val), 'Val indices must be identical for seed 42');
    assertDeep(Array.from(randomSplit1.test), Array.from(randomSplit2.test), 'Test indices must be identical for seed 42');

    // Verify all indices are disjoint and cover exactly 0..N-1
    const allIndices = new Set([...randomSplit1.fit, ...randomSplit1.val, ...randomSplit1.test]);
    assert.strictEqual(allIndices.size, N, `All ${N} indices must be unique across partitions`);

    console.log('   ✓ Split proportions (70% train / 30% test, 80% fit / 20% val) verified.');
    console.log('   ✓ Chronological monotonic ordering verified.');
    console.log('   ✓ Mulberry32 seed 42 determinism and disjoint coverage verified.');
  }

  // =========================================================================
  // TEST 6: JS-vs-Python Metrics Verification (sklearn reference fixtures with ties)
  // =========================================================================
  console.log('\nTest 6: JS-vs-Python Metrics Verification against sklearn fixtures (with ties)...');
  const metricsFixturePath = path.resolve(__dirname, 'fixtures/metrics_expected.json');
  if (!fs.existsSync(metricsFixturePath)) {
    throw new Error('metrics_expected.json not found. Run scripts/generate_metrics_fixture.py first.');
  }
  const metricsFixtures = JSON.parse(fs.readFileSync(metricsFixturePath, 'utf8'));

  for (const [caseName, data] of Object.entries(metricsFixtures)) {
    const prAuc = computePrAucJS(data.y_true, data.y_score);
    const rocAuc = computeRocAucJS(data.y_true, data.y_score);
    const prDiff = Math.abs(prAuc - data.pr_auc);
    const rocDiff = Math.abs(rocAuc - data.roc_auc);

    console.log(`   Case [${caseName}]: PR-AUC diff=${prDiff.toExponential(3)}, ROC-AUC diff=${rocDiff.toExponential(3)}`);
    assert.strictEqual(prDiff < 1e-9, true, `PR-AUC difference ${prDiff} must be < 1e-9 for ${caseName}`);
    assert.strictEqual(rocDiff < 1e-9, true, `ROC-AUC difference ${rocDiff} must be < 1e-9 for ${caseName}`);
  }
  console.log('   ✓ All JS metrics match sklearn reference fixtures within 1e-9.');

  // =========================================================================
  // TEST 7: Real Mode LightGBM Parity on Part A Sample (including NaNs)
  // =========================================================================
  console.log('\nTest 7: Parity Verification on Part A Sample (via Lambda handler with NaNs)...');
  const { execSync } = require('child_process');
  const parityGenScript = path.resolve(__dirname, '../scripts/generate_real_parity_data.py');
  execSync(`python "${parityGenScript}"`, { stdio: 'pipe' });

  const realParityDumpPath = path.resolve(__dirname, 'fixtures/real_parity_dump.json');
  const parityData = JSON.parse(fs.readFileSync(realParityDumpPath, 'utf8'));
  const model = parityData.model;
  const testRows = parityData.test_rows;
  const pyPreds = parityData.python_preds;

  prepareModelTrees(model);

  let maxDiff = 0.0;
  let sumDiff = 0.0;
  for (let i = 0; i < testRows.length; i++) {
    const jsProb = scoreRow(model, testRows[i]);
    const pyProb = pyPreds[i];
    const diff = Math.abs(jsProb - pyProb);
    if (diff > maxDiff) maxDiff = diff;
    sumDiff += diff;
  }
  const avgDiff = sumDiff / testRows.length;
  console.log(`   Evaluated ${testRows.length} rows (including rows with NaNs in non-NaN columns).`);
  console.log(`   Average absolute diff: ${avgDiff.toExponential(4)}`);
  console.log(`   Max absolute diff:     ${maxDiff.toExponential(4)}`);
  assert.strictEqual(maxDiff < 1e-9, true, `Parity check failed: maxDiff ${maxDiff} must be < 1e-9`);
  console.log(`   ✓ Real mode parity verified (max |diff| < 1e-9, exact LightGBM match with NaNs treated as 0).`);

  // =========================================================================
  // TEST 8: Real Mode Stage 6 Payload Builder
  // =========================================================================
  console.log('\nTest 8: Real Mode Payload Builder validation...');
  // Setup real mode state using creditcard sample
  const samplePath = path.resolve(__dirname, 'fixtures/real/creditcard_sample300.csv');
  const sampleContent = fs.readFileSync(samplePath, 'utf8');
  const sampleStream = makeStream(sampleContent, 4096);
  const parsed = await streamParseCSV(sampleStream);
  const detected = autoDetectColumns(parsed.headers, parsed.rawColumns, parsed.rawRowCount, parsed.nonNumericMap);
  const procRes = processRealColumns(parsed, detected.labelCol, detected.orderCol);
  const cleanRes = cleanRealDataset(procRes);
  const splitRes = splitRealDataset(cleanRes.rowCount, null, 42);

  state.realData = procRes;
  state.realClean = cleanRes;
  state.realSplits = {
    fit: splitRes.fit,
    val: splitRes.val,
    train: splitRes.train,
    test: splitRes.test
  };

  const { payload, payloadStr } = buildRealTrainPayload();

  // 1. Never send test slice
  assert.strictEqual(payload.test, undefined, 'Payload must NEVER contain a test slice');

  // 2. Class balanced fit slice: positives and negatives must be equal (1:1)
  const fitRows = payload.options.fit_rows;
  const fitY = payload.train.y.slice(0, fitRows);
  const fitPos = fitY.filter(y => y === 1).length;
  const fitNeg = fitY.filter(y => y === 0).length;
  assert.strictEqual(fitPos, fitNeg, `Fit slice must be class-balanced 1:1 (got ${fitPos} pos, ${fitNeg} neg)`);

  // 3. Validation slice follows fit slice
  const valRows = payload.train.y.length - fitRows;
  assert.strictEqual(valRows, splitRes.val.length, 'Validation slice rows must match validation partition');

  // 4. Budget under 4.5 MB
  assert.strictEqual(payloadStr.length < 4718592, true, 'Payload must be under 4.5 MB total');

  // 5. Number rounding to 6 significant digits
  for (let r = 0; r < Math.min(10, payload.train.X.length); r++) {
    for (const v of payload.train.X[r]) {
      if (v !== null) {
        const numStr = String(v);
        const sigDigits = numStr.replace(/^-?0*\.?0*/, '').replace(/e.*$/, '').replace('.', '');
        assert.strictEqual(sigDigits.length <= 6, true, `Number ${v} must have at most 6 significant digits (got ${sigDigits.length})`);
      }
    }
  }
  console.log(`   ✓ Payload size: ${(payloadStr.length / 1024).toFixed(1)} KB (< 4.5 MB budget).`);
  console.log(`   ✓ Fit slice class-balanced 1:1 (${fitPos} pos, ${fitNeg} neg).`);
  console.log(`   ✓ Test slice omitted (NEVER sent).`);
  console.log(`   ✓ Numeric features rounded to 6 significant digits.`);

  // -------------------------------------------------------------
  // Test 9: Stage 7 and Dashboard Held-Out Replay Logic
  // - Action counts sum to the test row count
  // - Gain curve ends at 100% / 100%
  // - Top-50 table is sorted by score descending
  // -------------------------------------------------------------
  console.log('\n--- Test 9: Stage 7 and Dashboard Replay Metrics & Ordering ---');

  // Generate 120 pseudo scores and labels with varied distributions
  const testScores = new Float64Array(120);
  const testLabels = new Uint8Array(120);
  for (let i = 0; i < 120; i++) {
    testScores[i] = +(((i * 73) % 1000) / 1000).toFixed(4); // values in [0, 0.999]
    testLabels[i] = (i % 7 === 0 || testScores[i] > 0.85) ? 1 : 0;
  }

  const th1 = { review: 0.35, decision: 0.50, block: 0.75 };
  const replay1 = computeRealReplay(testScores, testLabels, null, null, null, ['V1', 'V2', 'V3'], th1);

  // Assertion 1: Action counts sum to the test row count
  const sumCounts1 = replay1.counts.Allow.count + replay1.counts.Review.count + replay1.counts.Block.count;
  assert.strictEqual(sumCounts1, testScores.length, `Action counts must sum to test rows (got ${sumCounts1}, expected ${testScores.length})`);
  assert.strictEqual(replay1.counts.Allow.frauds + replay1.counts.Review.frauds + replay1.counts.Block.frauds, replay1.totalFrauds, 'Sum of frauds per action must equal total frauds');
  assert.strictEqual(replay1.counts.Allow.legit + replay1.counts.Review.legit + replay1.counts.Block.legit, replay1.totalLegit, 'Sum of legit per action must equal total legit');
  console.log(`   ✓ Action counts sum to test row count: ${replay1.counts.Allow.count} Allow + ${replay1.counts.Review.count} Review + ${replay1.counts.Block.count} Block = ${sumCounts1} (${testScores.length} test rows).`);

  // Assertion 2: Gain curve starts at (0, 0) and ends at exactly 100% / 100% (1.0, 1.0)
  const gainCurve1 = computeGainCurve(testScores, testLabels);
  assert.strictEqual(gainCurve1[0].x, 0, 'Gain curve must start at x=0');
  assert.strictEqual(gainCurve1[0].y, 0, 'Gain curve must start at y=0');
  const lastPoint1 = gainCurve1[gainCurve1.length - 1];
  assert.strictEqual(lastPoint1.x, 1, `Gain curve must end at x=1.0 (100%), got ${lastPoint1.x}`);
  assert.strictEqual(lastPoint1.y, 1, `Gain curve must end at y=1.0 (100%), got ${lastPoint1.y}`);
  // Verify monotonic increase
  for (let k = 1; k < gainCurve1.length; k++) {
    assert.strictEqual(gainCurve1[k].x >= gainCurve1[k - 1].x, true, `Gain curve x must be monotonically non-decreasing at step ${k}`);
    assert.strictEqual(gainCurve1[k].y >= gainCurve1[k - 1].y, true, `Gain curve y must be monotonically non-decreasing at step ${k}`);
  }
  console.log(`   ✓ Cumulative gain curve has ${gainCurve1.length} points, starting at (0, 0) and ending at exactly (${(lastPoint1.x * 100).toFixed(0)}%, ${(lastPoint1.y * 100).toFixed(0)}%).`);

  // Assertion 3: Top-50 table is sorted by score descending
  assert.strictEqual(replay1.top50.length, 50, `Top-50 table must contain 50 rows (got ${replay1.top50.length})`);
  for (let j = 0; j < replay1.top50.length - 1; j++) {
    assert.strictEqual(replay1.top50[j].score >= replay1.top50[j + 1].score, true,
      `Top-50 row ${j} (score: ${replay1.top50[j].score}) must be >= row ${j + 1} (score: ${replay1.top50[j + 1].score})`);
  }
  console.log(`   ✓ Top-50 table is strictly sorted by score descending: #${replay1.top50[0].rank} score ${replay1.top50[0].score.toFixed(4)} down to #${replay1.top50[49].rank} score ${replay1.top50[49].score.toFixed(4)}.`);

  // Test edge case: review == block (thresholds equal)
  const thEqual = { review: 0.60, decision: 0.60, block: 0.60 };
  const replayEqual = computeRealReplay(testScores, testLabels, null, null, null, ['V1', 'V2', 'V3'], thEqual);
  assert.strictEqual(replayEqual.isThresholdsEqual, true, 'isThresholdsEqual must be true when review == block');
  assert.strictEqual(replayEqual.counts.Review.count, 0, 'Review count must be 0 when review == block');
  assert.strictEqual(replayEqual.counts.Allow.count + replayEqual.counts.Block.count, testScores.length, 'Allow + Block must equal test count when Review is empty');
  console.log(`   ✓ Equal thresholds (review == block = 0.60) correctly flags empty Review zone (Review count = 0, Allow + Block = ${testScores.length}).`);

  // Test on real fixture partition test slice
  const fixtureTestCount = splitRes.test.length;
  const fixtureFakeScores = new Float64Array(fixtureTestCount);
  const fixtureLabels = new Uint8Array(fixtureTestCount);
  const rawLabels = procRes.rawColumns.get(procRes.labelCol);
  for (let idx = 0; idx < fixtureTestCount; idx++) {
    const origIdx = cleanRes.validIndices[splitRes.test[idx]];
    fixtureLabels[idx] = rawLabels[origIdx];
    fixtureFakeScores[idx] = fixtureLabels[idx] === 1 ? 0.92 : (idx % 10 === 0 ? 0.76 : 0.15);
  }
  const fixtureReplay = computeRealReplay(
    fixtureFakeScores,
    fixtureLabels,
    cleanRes.validIndices,
    splitRes.test,
    procRes.rawColumns,
    cleanRes.featureCols,
    { review: 0.40, decision: 0.50, block: 0.70 }
  );
  assert.strictEqual(
    fixtureReplay.counts.Allow.count + fixtureReplay.counts.Review.count + fixtureReplay.counts.Block.count,
    fixtureTestCount,
    'Fixture test slice action counts must sum to test count'
  );
  const fixtureGain = computeGainCurve(fixtureFakeScores, fixtureLabels);
  assert.strictEqual(fixtureGain[fixtureGain.length - 1].x, 1);
  assert.strictEqual(fixtureGain[fixtureGain.length - 1].y, 1);
  for (let j = 0; j < fixtureReplay.top50.length - 1; j++) {
    assert.strictEqual(fixtureReplay.top50[j].score >= fixtureReplay.top50[j + 1].score, true);
  }
  console.log(`   ✓ Verified on real creditcard_sample300 fixture (${fixtureTestCount} test rows): counts sum, gain curve ends at (100%, 100%), top-50 sorted descending.`);

  // Test 10: Live Replay Stream Simulation Counter Invariants
  console.log('\nTest 10: Live Replay Stream Simulation Counter Invariants');
  const simResult = simulateRealStreamReplay(
    fixtureFakeScores,
    fixtureLabels,
    { review: 0.40, decision: 0.50, block: 0.70 }
  );
  assert.strictEqual(simResult.counts.replayed, fixtureTestCount, 'Counters replayed must equal number of test rows at end of replay');
  assert.strictEqual(
    simResult.counts.Allow + simResult.counts.Review + simResult.counts.Block,
    simResult.counts.replayed,
    'Allow + Review + Block counters must sum to total replayed rows'
  );
  assert.strictEqual(simResult.counts.Allow, fixtureReplay.counts.Allow.count, 'Allow counter matches static replay count');
  assert.strictEqual(simResult.counts.Review, fixtureReplay.counts.Review.count, 'Review counter matches static replay count');
  assert.strictEqual(simResult.counts.Block, fixtureReplay.counts.Block.count, 'Block counter matches static replay count');
  assert.strictEqual(simResult.counts.fraudsCaught, fixtureReplay.counts.Block.frauds, 'Frauds caught among Block counter matches static replay');
  assert.strictEqual(simResult.rows.length, fixtureTestCount, 'Total simulated rows array length matches test count');
  console.log(`   ✓ Stream counters equal number of replayed rows at the end: ${simResult.counts.replayed} replayed = ${simResult.counts.Allow} Allow + ${simResult.counts.Review} Review + ${simResult.counts.Block} Block (${simResult.counts.fraudsCaught} frauds caught).`);

  console.log('\n=== ALL REAL DATASET MODE UNIT TESTS PASSED ===');
}

runTests().catch(err => {
  console.error('\n❌ Real mode test failure:', err);
  process.exit(1);
});
