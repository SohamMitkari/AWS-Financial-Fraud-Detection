/**
 * Causal Feature Engineering Verification Script (Node.js)
 * Tests that modifying, deleting, or appending rows AFTER a cut point k
 * strictly NEVER alters the feature vector of any row in 0..k.
 * Uses the REAL extractCausalFeatures and generate from frontend/index.html via _extract.js.
 */

const assert = require('assert');
const { loadFrontendFunctions } = require('./_extract');

console.log('1. Loading REAL extractCausalFeatures and generate from frontend/index.html...');
const { extractCausalFeatures, generate } = loadFrontendFunctions(['extractCausalFeatures', 'generate']);

if (typeof extractCausalFeatures !== 'function' || typeof generate !== 'function') {
  throw new Error('Failed to load extractCausalFeatures or generate');
}

console.log('2. Generating 5,000 baseline records (seed: 20261006)...');
const NUM_ROWS = 5000;
const SEED = 20261006;
const rows = generate(NUM_ROWS, SEED);

if (rows.length !== NUM_ROWS) {
  throw new Error(`Expected ${NUM_ROWS} rows, got ${rows.length}`);
}

console.log('3. Extracting baseline causal feature vectors...');
const baseline = extractCausalFeatures(rows);
const baselineX = baseline.X;

// Select 50 deterministic random cut points k spread across the stream
const cutPoints = [];
let rngState = 20261022;
function randInt(min, max) {
  rngState = (rngState * 9301 + 49297) % 233280;
  return Math.floor(min + (rngState / 233280) * (max - min));
}

while (cutPoints.length < 50) {
  const k = randInt(5, NUM_ROWS - 50);
  if (!cutPoints.includes(k)) cutPoints.push(k);
}
cutPoints.sort((a, b) => a - b);

console.log('4. Testing 50 cut points across mutation, deletion, and append operations...');

for (let idx = 0; idx < cutPoints.length; idx++) {
  const k = cutPoints[idx];

  // Test 1: Mutate rows after k
  const mutated = rows.map((r, i) => {
    if (i > k) {
      return {
        ...r,
        amount: (r.amount || 0) * 3 + 750,
        oldbalanceOrg: 987654.32,
        newbalanceOrig: 0,
        nameOrig: 'C_MUTATED_' + i,
        nameDest: 'M_MUTATED_' + i,
        type: r.type === 'TRANSFER' ? 'PAYMENT' : 'TRANSFER',
        isFraud: r.isFraud === 1 ? 0 : 1
      };
    }
    return r;
  });
  const mutRes = extractCausalFeatures(mutated);
  for (let i = 0; i <= k; i++) {
    assert.deepStrictEqual(
      mutRes.X[i],
      baselineX[i],
      `Causal leakage violation: row ${i} feature vector altered after mutating rows > k (k=${k})`
    );
  }

  // Test 2: Delete rows after k
  const deleted = rows.slice(0, k + 1);
  const delRes = extractCausalFeatures(deleted);
  for (let i = 0; i <= k; i++) {
    assert.deepStrictEqual(
      delRes.X[i],
      baselineX[i],
      `Causal leakage violation: row ${i} feature vector altered after deleting rows > k (k=${k})`
    );
  }

  // Test 3: Append rows after k
  const appended = rows.slice(0, k + 1);
  for (let a = 1; a <= 20; a++) {
    appended.push({
      step: Math.floor((k + a) / 24) + 1,
      type: a % 2 === 0 ? 'CASH_OUT' : 'TRANSFER',
      amount: 15000 + a * 100,
      nameOrig: 'C_APPEND_' + a,
      oldbalanceOrg: 50000,
      newbalanceOrig: 35000,
      nameDest: 'M_APPEND_' + a,
      oldbalanceDest: 0,
      newbalanceDest: 15000,
      isFraud: 1
    });
  }
  const appRes = extractCausalFeatures(appended);
  for (let i = 0; i <= k; i++) {
    assert.deepStrictEqual(
      appRes.X[i],
      baselineX[i],
      `Causal leakage violation: row ${i} feature vector altered after appending rows > k (k=${k})`
    );
  }
}

console.log('✓ Causal feature engineering verification PASSED: zero lookahead leakage detected.');
console.log('✓ Evaluated 50 random cut points on 5,000 rows across mutate, delete, and append operations (all rows 0..k strictly invariant).');
