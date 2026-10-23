/**
 * Export Sample Payload
 * Generates transactions using the real frontend generator,
 * performs Stage 2 cleaning, extracts causal features,
 * splits chronologically 70/30, applies trainModelStage sampling rules,
 * and writes backend/tests/sample_payload.json.
 *
 * Usage:
 *   node backend/tests/export_payload.js [seed]
 */

const fs = require('fs');
const path = require('path');
const { loadFrontendFunctions } = require('./_extract');

const seed = process.argv[2] ? parseInt(process.argv[2], 10) : 20261006;
console.log(`Generating payload with seed: ${seed}...`);

const { generate, extractCausalFeatures, expectedBalances, FEATURE_NAMES } = loadFrontendFunctions([
  'generate',
  'extractCausalFeatures',
  'expectedBalances',
  'FEATURE_NAMES'
]);

// 1. Deterministic generation of 50,000 rows
const raw = generate(50000, seed);
console.log(`Generated ${raw.length} raw transactions.`);

// 2. Stage 2: Integrity & Balance Continuity Validation
const seen = new Set();
const cleaned = [];
let nullCount = 0;
let negCount = 0;
let dupCount = 0;
let balanceErrors = 0;

for (const z of raw) {
  let hasNull = false;
  let hasNeg = false;
  ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud'].forEach(k => {
    if (z[k] == null || z[k] === '') hasNull = true;
  });
  if (hasNull) nullCount++;

  const amt = +z.amount || 0;
  if (amt < 0 || +z.oldbalanceOrg < 0 || +z.newbalanceOrig < 0 || +z.oldbalanceDest < 0 || +z.newbalanceDest < 0) {
    negCount++;
    hasNeg = true;
  }

  const safeAmt = Math.abs(amt);
  const oldO = Math.max(0, Number.isFinite(+z.oldbalanceOrg) ? +z.oldbalanceOrg : 0);
  const newO = Math.max(0, Number.isFinite(+z.newbalanceOrig) ? +z.newbalanceOrig : 0);
  const oldD = Math.max(0, Number.isFinite(+z.oldbalanceDest) ? +z.oldbalanceDest : 0);
  const newD = Math.max(0, Number.isFinite(+z.newbalanceDest) ? +z.newbalanceDest : 0);

  const { expectedNewO, expectedNewD } = expectedBalances(z.type, oldO, oldD, safeAmt);
  const origErr = Math.abs(expectedNewO - newO) > 0.05;
  const destErr = Math.abs(expectedNewD - newD) > 0.05;
  if (origErr || destErr) balanceErrors++;

  const key = [z.step, z.type, z.nameOrig, z.nameDest, safeAmt].join('|');
  if (seen.has(key)) {
    dupCount++;
    continue;
  }
  seen.add(key);

  cleaned.push({
    step: +z.step || 1,
    type: z.type || 'PAYMENT',
    amount: safeAmt,
    nameOrig: z.nameOrig || 'C0',
    oldbalanceOrg: oldO,
    newbalanceOrig: newO,
    nameDest: z.nameDest || 'M0',
    oldbalanceDest: oldD,
    newbalanceDest: newD,
    isFraud: +z.isFraud === 1 ? 1 : 0
  });
}

console.log(`Stage 2 Audit: nulls=${nullCount}, negs=${negCount}, dups=${dupCount}, balanceErrors=${balanceErrors}. Cleaned: ${cleaned.length} rows.`);

// Ensure chronological order
cleaned.sort((a, b) => a.step - b.step);

// 3. Strictly Causal Feature Extraction
const { X, y } = extractCausalFeatures(cleaned);
console.log(`Extracted causal features for ${X.length} records (${FEATURE_NAMES.length} features).`);

// 4. Chronological 70/30 Split
const trainCount = Math.floor(X.length * 0.7);
const X_tr_full = X.slice(0, trainCount);
const y_tr_full = y.slice(0, trainCount);
const X_te = X.slice(trainCount);
const y_te = y.slice(trainCount);
console.log(`Chronological split (70/30): ${X_tr_full.length} train rows, ${X_te.length} test rows.`);

function buildTrainPayload(trainX, trainY) {
  return {
    feature_names: FEATURE_NAMES,
    train: { X: trainX, y: trainY },
    test: { X: X_te, y: y_te },
    options: {
      num_boost_round: 300,
      learning_rate: 0.05,
      num_leaves: 15,
      max_depth: 6,
      min_data_in_leaf: 50,
      seed: 42
    },
    meta: {
      split: '70/30',
      n_raw: raw.length,
      n_clean: cleaned.length
    }
  };
}

// 5. Apply trainModelStage sampling rule
let X_tr = X_tr_full;
let y_tr = y_tr_full;
let requestPayload = buildTrainPayload(X_tr, y_tr);
let payloadBodyStr = JSON.stringify(requestPayload);

if (payloadBodyStr.length <= 5000000) {
  console.log(`Payload size: ${payloadBodyStr.length} chars (<= 5,000,000): keeping full train partition (${X_tr.length} rows) without undersampling.`);
} else {
  console.log(`Payload size: ${payloadBodyStr.length} chars (> 5,000,000): undersampling fit slice (first 80%)...`);
  const n_train_total = X_tr_full.length;
  const fitSplit = Math.floor(n_train_total * 0.80);

  const X_val = X_tr_full.slice(fitSplit);
  const y_val = y_tr_full.slice(fitSplit);

  const fitFraudIdx = [];
  const fitLegitIdx = [];
  for (let i = 0; i < fitSplit; i++) {
    if (y_tr_full[i] === 1) fitFraudIdx.push(i);
    else fitLegitIdx.push(i);
  }

  const minX = fitFraudIdx.map(idx => X_tr_full[idx]).concat(X_val);
  const minY = fitFraudIdx.map(idx => y_tr_full[idx]).concat(y_val);
  const zeroLen = JSON.stringify(buildTrainPayload(minX, minY)).length;

  const charsForFitLegit = payloadBodyStr.length - zeroLen;
  const avgCharsPerLegit = charsForFitLegit / Math.max(1, fitLegitIdx.length);

  let allowedLegit = Math.max(0, Math.floor((4950000 - zeroLen) / avgCharsPerLegit));
  allowedLegit = Math.min(allowedLegit, fitLegitIdx.length);

  const sampleFit = (k) => {
    const step = fitLegitIdx.length / k;
    const chosenLegit = [];
    for (let j = 0; j < k; j++) {
      chosenLegit.push(fitLegitIdx[Math.floor(j * step)]);
    }
    const allFitChosen = [...fitFraudIdx, ...chosenLegit].sort((a, b) => a - b);
    const subX = allFitChosen.map(idx => X_tr_full[idx]).concat(X_val);
    const subY = allFitChosen.map(idx => y_tr_full[idx]).concat(y_val);
    return { subX, subY };
  };

  let sampled = sampleFit(allowedLegit);
  requestPayload = buildTrainPayload(sampled.subX, sampled.subY);
  payloadBodyStr = JSON.stringify(requestPayload);

  while (payloadBodyStr.length > 5000000 && allowedLegit > 100) {
    allowedLegit -= 200;
    sampled = sampleFit(allowedLegit);
    requestPayload = buildTrainPayload(sampled.subX, sampled.subY);
    payloadBodyStr = JSON.stringify(requestPayload);
  }

  X_tr = sampled.subX;
  y_tr = sampled.subY;
  console.log(`Undersampled fit slice: kept ${allowedLegit} legitimate rows. Total train rows: ${X_tr.length}.`);
}

// 6. Write output
const outputPath = process.argv[3] ? path.resolve(process.argv[3]) : path.resolve(__dirname, 'sample_payload.json');
fs.writeFileSync(outputPath, payloadBodyStr);
console.log(`Successfully written ${outputPath} (${(payloadBodyStr.length / 1024 / 1024).toFixed(2)} MB).`);
