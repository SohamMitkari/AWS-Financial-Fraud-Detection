#!/usr/bin/env node
/**
 * Headless Evaluation Runner for Real Datasets (Part D)
 * Reuses the exact JS functions from frontend/index.html (streamParseCSV, autoDetectColumns,
 * processRealColumns, cleanRealDataset, splitRealDataset, buildRealTrainPayload, scoreRow,
 * computePrAucJS, computeRocAucJS) and the local Python Lambda handler to evaluate all
 * available real fraud datasets.
 *
 * Writes results to results/real_eval.json.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { execSync } = require('child_process');

const rootDir = path.resolve(__dirname, '..');

// 1. Extract and load frontend functions from frontend/index.html
function loadFrontendFunctions() {
  const htmlPath = path.join(rootDir, 'frontend/index.html');
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');
  const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>[\s\S]*?$/);
  if (!scriptMatch) throw new Error('Could not find <script> in frontend/index.html');
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
    'splitRealDataset',
    'buildRealTrainPayload',
    'flattenTree',
    'prepareModelTrees',
    'scoreRow',
    'sigmoidJS',
    'computePrAucJS',
    'computeRocAucJS'
  ];

  const snippets = [];
  for (const name of fnNames) {
    const fnCode = extractFunction(scriptCode, name);
    if (!fnCode) throw new Error(`Could not extract function ${name} from frontend/index.html`);
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

// 2. File streaming helper matching the web stream interface
function createNodeFileStream(filePath) {
  const stream = fs.createReadStream(filePath, { highWaterMark: 64 * 1024 });
  const iterator = stream[Symbol.asyncIterator]();
  return {
    getReader() {
      return {
        async read() {
          const { done, value } = await iterator.next();
          return {
            done,
            value: value ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength) : undefined
          };
        },
        async cancel() {
          stream.destroy();
        }
      };
    }
  };
}

const DATASETS = [
  {
    key: 'creditcard',
    displayName: 'Credit Card Fraud',
    rawFiles: ['dataset/creditcard/creditcard.csv'],
    stdFile: 'dataset/creditcard/creditcard_std.csv',
    prepScript: 'scripts/prep_creditcard.py'
  },
  {
    key: 'elliptic',
    displayName: 'Elliptic Bitcoin',
    rawFiles: [
      'dataset/elliptic/elliptic_txs_features.csv',
      'dataset/elliptic/elliptic_txs_classes.csv'
    ],
    stdFile: 'dataset/elliptic/elliptic_std.csv',
    prepScript: 'scripts/prep_elliptic.py'
  },
  {
    key: 'ethereum',
    displayName: 'Ethereum Fraud',
    rawFiles: ['dataset/ethereum/transaction_dataset.csv'],
    stdFile: 'dataset/ethereum/ethereum_std.csv',
    prepScript: 'scripts/prep_ethereum.py'
  },
  {
    key: 'bitcoinheist',
    displayName: 'BitcoinHeist Ransomware',
    rawFiles: ['dataset/bitcoinheist/BitcoinHeistData.csv'],
    stdFile: 'dataset/bitcoinheist/bitcoinheist_std.csv',
    prepScript: 'scripts/prep_bitcoinheist.py'
  },
  {
    key: 'taiwan',
    displayName: 'Taiwan Credit Default',
    rawFiles: ['dataset/taiwan/UCI_Credit_Card.csv'],
    stdFile: 'dataset/taiwan/taiwan_std.csv',
    prepScript: 'scripts/prep_taiwan.py'
  }
];

async function main() {
  console.log('========================================================================');
  console.log('HEADLESS REAL DATASET EVALUATION RUNNER');
  console.log('========================================================================');

  const sandbox = loadFrontendFunctions();
  const results = {};

  for (const ds of DATASETS) {
    console.log(`\nEvaluating: ${ds.displayName} (${ds.key})...`);

    // Check if raw files exist
    const rawExists = ds.rawFiles.every(rf => fs.existsSync(path.join(rootDir, rf)));
    if (!rawExists) {
      console.log(`   [SKIP] Raw file(s) missing: ${ds.rawFiles.join(', ')} -> marked 'not run'`);
      results[ds.key] = {
        name: ds.displayName,
        status: 'not run',
        reason: 'Raw dataset file not present in workspace'
      };
      continue;
    }

    // Ensure standard file exists
    const stdPath = path.join(rootDir, ds.stdFile);
    if (!fs.existsSync(stdPath)) {
      console.log(`   Standard CSV missing; running prep script: ${ds.prepScript}...`);
      execSync(`python "${path.join(rootDir, ds.prepScript)}"`, { stdio: 'inherit' });
    }

    const t0 = Date.now();

    // 1. Parse CSV with streaming reader
    console.log(`   1. Streaming & parsing ${ds.stdFile}...`);
    const stream = createNodeFileStream(stdPath);
    const parsed = await sandbox.streamParseCSV(stream);
    console.log(`      Parsed ${parsed.rawRowCount.toLocaleString()} raw rows, ${parsed.headers.length} columns.`);

    // 2. Auto-detect columns
    const detected = sandbox.autoDetectColumns(parsed.headers, parsed.rawColumns, parsed.rawRowCount, parsed.nonNumericMap);
    console.log(`      Auto-detected: label='${detected.labelCol}', order='${detected.orderCol}'`);

    // 3. Process & clean
    const proc = sandbox.processRealColumns(parsed, detected.labelCol, detected.orderCol);
    const cleaned = sandbox.cleanRealDataset(proc);
    console.log(`      Cleaned: ${cleaned.rowCount.toLocaleString()} valid rows (${proc.fraudCount.toLocaleString()} frauds, ${(proc.fraudCount / proc.rowCount * 100).toFixed(2)}% prevalence), ${cleaned.featureCols.length} features.`);

    // 4. Split dataset (chronological if order col present, else seeded mulberry32(42))
    const N = cleaned.rowCount;
    let orderSubarray = null;
    if (detected.orderCol && detected.orderCol !== '__none__') {
      const rawOrder = proc.rawColumns.get(detected.orderCol);
      orderSubarray = new Float64Array(N);
      for (let i = 0; i < N; i++) orderSubarray[i] = rawOrder[cleaned.validIndices[i]];
    }

    const splitRes = sandbox.splitRealDataset(N, orderSubarray, 42);
    console.log(`      Split mode: ${splitRes.splitMode} (Fit: ${splitRes.fit.length}, Val: ${splitRes.val.length}, Test: ${splitRes.test.length})`);

    // Setup state for payload builder
    sandbox.state.realData = proc;
    sandbox.state.realClean = cleaned;
    sandbox.state.realSplits = splitRes;

    // 5. Build Stage 6 payload (1:1 class balanced fit, natural val slice subsampled under 4.5 MB budget, test never sent)
    console.log('   2. Building transmission payload...');
    let payloadRes;
    try {
      payloadRes = sandbox.buildRealTrainPayload();
    } catch (err) {
      console.log(`      ⚠️ Payload builder error: ${err.message}`);
      results[ds.key] = {
        name: ds.displayName,
        status: 'exceeds_budget',
        error: err.message,
        split_type: splitRes.splitMode,
        rows: proc.rowCount,
        positives: proc.fraudCount,
        prevalence: +(proc.fraudCount / proc.rowCount).toFixed(6),
        prevalence_pct: +((proc.fraudCount / proc.rowCount) * 100).toFixed(4),
        features: cleaned.featureCols.length,
        rows_sent: null,
        payload_mb: null,
        pr_auc: null,
        roc_auc: null,
        precision: null,
        recall: null,
        f1: null,
        confusion_matrix: null
      };
      continue;
    }
    const { payload, payloadStr } = payloadRes;
    const info = sandbox.state.realPayloadInfo;
    console.log(`      Fit slice: ${info.fitRows} rows (${info.fitPos} pos / ${info.fitNeg} neg, 1:1 balanced)`);
    console.log(`      Validation slice: ${info.valRows} rows ${info.valSubsampled ? `(subsampled from ${info.valTotal})` : '(100% natural)'}`);
    console.log(`      Payload size: ${info.payloadMB} MB (< 4.5 MB budget). Test slice: OMITTED.`);

    // 6. Train model via local Python Lambda handler
    console.log('   3. Training LightGBM in local Lambda handler...');
    const tempPayloadPath = path.join(rootDir, 'scripts', `temp_${ds.key}_payload.json`);
    const tempResultPath = path.join(rootDir, 'scripts', `temp_${ds.key}_result.json`);
    fs.writeFileSync(tempPayloadPath, payloadStr, 'utf8');

    const helperScript = path.join(rootDir, 'scripts/train_local_helper.py');
    execSync(`python "${helperScript}" "${tempPayloadPath}" "${tempResultPath}"`, { stdio: 'inherit' });

    const trainResult = JSON.parse(fs.readFileSync(tempResultPath, 'utf8'));

    // Clean up temporary files
    try { fs.unlinkSync(tempPayloadPath); } catch (_) {}
    try { fs.unlinkSync(tempResultPath); } catch (_) {}

    const model = trainResult.model || trainResult;
    if (trainResult.thresholds && !model.thresholds) model.thresholds = trainResult.thresholds;
    sandbox.prepareModelTrees(model);

    // 7. Score FULL test partition in pure JS
    console.log(`   4. Scoring full test partition (${splitRes.test.length.toLocaleString()} rows) in pure JS...`);
    const d = cleaned.featureCols.length;
    const validIndices = cleaned.validIndices;
    const labelArr = proc.rawColumns.get(proc.labelCol);
    const featColsData = cleaned.featureCols.map(c => proc.rawColumns.get(c));
    const testIndices = splitRes.test;

    const testScores = new Float64Array(testIndices.length);
    const testLabels = new Int32Array(testIndices.length);

    for (let i = 0; i < testIndices.length; i++) {
      const cleanIdx = testIndices[i];
      const origIdx = validIndices[cleanIdx];
      const row = new Array(d);
      for (let j = 0; j < d; j++) {
        row[j] = featColsData[j][origIdx];
      }
      testScores[i] = sandbox.scoreRow(model, row);
      testLabels[i] = labelArr[origIdx];
    }

    // 8. Compute metrics on test partition
    const prAuc = sandbox.computePrAucJS(Array.from(testLabels), Array.from(testScores));
    const rocAuc = sandbox.computeRocAucJS(Array.from(testLabels), Array.from(testScores));
    const decTh = model.thresholds?.decision ?? 0.50;

    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < testLabels.length; i++) {
      const pred = testScores[i] >= decTh ? 1 : 0;
      const actual = testLabels[i];
      if (pred === 1 && actual === 1) tp++;
      else if (pred === 1 && actual === 0) fp++;
      else if (pred === 0 && actual === 1) fn++;
      else tn++;
    }

    const precision = tp / Math.max(1, tp + fp);
    const recall = tp / Math.max(1, tp + fn);
    const accuracy = (tp + tn) / Math.max(1, testLabels.length);
    const f1 = (precision + recall > 0) ? (2 * precision * recall) / (precision + recall) : 0;

    const elapsedSec = ((Date.now() - t0) / 1000).toFixed(1);

    console.log(`      ✓ PR-AUC (Headline): ${prAuc.toFixed(4)}`);
    console.log(`      ✓ ROC-AUC:          ${rocAuc.toFixed(4)}`);
    console.log(`      ✓ Precision/Recall:  ${(precision * 100).toFixed(2)}% / ${(recall * 100).toFixed(2)}% (F1: ${(f1 * 100).toFixed(2)}%)`);
    console.log(`      ✓ Confusion Matrix:  TP=${tp}, FP=${fp}, FN=${fn}, TN=${tn} (@ decision cut ${decTh})`);
    console.log(`      ✓ Completed in ${elapsedSec}s.`);

    results[ds.key] = {
      name: ds.displayName,
      status: 'completed',
      split_type: splitRes.splitMode,
      rows: proc.rowCount,
      positives: proc.fraudCount,
      prevalence: +(proc.fraudCount / proc.rowCount).toFixed(6),
      prevalence_pct: +((proc.fraudCount / proc.rowCount) * 100).toFixed(4),
      rows_sent: {
        fit: info.fitRows,
        fit_positives: info.fitPos,
        fit_negatives: info.fitNeg,
        val: info.valRows,
        val_total: info.valTotal,
        val_subsampled: info.valSubsampled,
        total_train_sent: info.fitRows + info.valRows
      },
      test_rows: testIndices.length,
      payload_mb: +info.payloadMB,
      pr_auc: +prAuc.toFixed(4),
      roc_auc: +rocAuc.toFixed(4),
      precision: +precision.toFixed(4),
      recall: +recall.toFixed(4),
      f1: +f1.toFixed(4),
      accuracy: +accuracy.toFixed(4),
      decision_threshold: decTh,
      thresholds: model.thresholds,
      confusion_matrix: { tp, fp, fn, tn },
      elapsed_sec: +elapsedSec
    };
  }

  // 9. Write results/real_eval.json
  const outDir = path.join(rootDir, 'results');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, 'real_eval.json');
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2), 'utf8');

  console.log('\n========================================================================');
  console.log(`EVALUATION COMPLETE. Results saved to: ${outPath}`);
  console.log('========================================================================\n');

  console.log('BENCHMARK SUMMARY TABLE:');
  console.log('---------------------------------------------------------------------------------------------------------------------------');
  console.log('Dataset         | Split Type     | Rows    | Positives | Prev (%) | Train Sent | Payl (MB) | PR-AUC | ROC-AUC | F1     | Recall ');
  console.log('---------------------------------------------------------------------------------------------------------------------------');
  for (const [k, r] of Object.entries(results)) {
    if (r.status === 'not run') {
      console.log(`${k.padEnd(15)} | NOT RUN (raw file missing)`);
    } else if (r.status === 'exceeds_budget') {
      const name = k.padEnd(15);
      const split = r.split_type.padEnd(14);
      const rows = String(r.rows).padStart(7);
      const pos = String(r.positives).padStart(9);
      const prev = (r.prevalence_pct.toFixed(2) + '%').padStart(8);
      console.log(`${name} | ${split} | ${rows} | ${pos} | ${prev} | EXCEEDS 4.5 MB BUDGET (fit slice > 4.5 MB for ${r.features} features)`);
    } else {
      const name = k.padEnd(15);
      const split = r.split_type.padEnd(14);
      const rows = String(r.rows).padStart(7);
      const pos = String(r.positives).padStart(9);
      const prev = (r.prevalence_pct.toFixed(2) + '%').padStart(8);
      const sent = String(r.rows_sent.total_train_sent).padStart(10);
      const mb = (r.payload_mb.toFixed(2) + ' MB').padStart(9);
      const pr = r.pr_auc.toFixed(4).padStart(6);
      const roc = r.roc_auc.toFixed(4).padStart(7);
      const f1 = (r.f1 * 100).toFixed(1) + '%';
      const rec = (r.recall * 100).toFixed(1) + '%';
      console.log(`${name} | ${split} | ${rows} | ${pos} | ${prev} | ${sent} | ${mb} | ${pr} | ${roc} | ${f1.padStart(6)} | ${rec.padStart(6)}`);
    }
  }
  console.log('---------------------------------------------------------------------------------------------------------------------------');
}

main().catch(err => {
  console.error('\n❌ Execution error in run_real_eval:', err);
  process.exit(1);
});
