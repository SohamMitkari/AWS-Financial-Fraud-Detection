/**
 * Automated JS/Python Parity Verification
 * Trains LightGBM in Python on backend/tests/sample_payload.json,
 * exports browser-format JSON model and 1,000 test row predictions,
 * extracts the REAL scoreRow function from frontend/index.html using _extract.js,
 * and asserts that max |difference| < 1e-9.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { loadFrontendFunctions } = require('./_extract');

const rootDir = path.resolve(__dirname, '../..');
const generateScriptPath = path.join(rootDir, 'backend/tests/generate_parity_data.py');
const tempDumpPath = path.join(rootDir, 'backend/tests/temp_parity_data.json');

console.log('1. Training LightGBM in Python on sample_payload.json and generating 1,000 test predictions...');
execSync(`python "${generateScriptPath}"`, { stdio: 'inherit' });

console.log('2. Extracting real tree scorer functions from frontend/index.html via _extract helper...');
const { flattenTree, prepareModelTrees, scoreRow, sigmoidJS } = loadFrontendFunctions([
  'flattenTree',
  'prepareModelTrees',
  'scoreRow',
  'sigmoidJS'
]);

if (typeof scoreRow !== 'function') {
  throw new Error('Extracted scoreRow is not a function');
}

console.log('3. Running JS tree scorer on 1,000 test rows and comparing with Python LightGBM...');
const dumpData = JSON.parse(fs.readFileSync(tempDumpPath, 'utf-8'));
const model = dumpData.model;
const testRows = dumpData.test_rows;
const pythonPreds = dumpData.python_preds;

prepareModelTrees(model);

let maxDiff = 0.0;
let sumDiff = 0.0;

for (let i = 0; i < testRows.length; i++) {
  const jsProb = scoreRow(model, testRows[i]);
  const pyProb = pythonPreds[i];
  const diff = Math.abs(jsProb - pyProb);
  if (diff > maxDiff) maxDiff = diff;
  sumDiff += diff;
}

const avgDiff = sumDiff / testRows.length;

console.log(`Evaluated ${testRows.length} rows.`);
console.log(`Average absolute difference: ${avgDiff.toExponential(4)}`);
console.log(`Max absolute difference:     ${maxDiff.toExponential(4)}`);

// Clean up temp file
try { fs.unlinkSync(tempDumpPath); } catch (_) {}

// Assert pass criterion
if (maxDiff >= 1e-9) {
  throw new Error(`PARITY FAILED: max difference ${maxDiff} >= 1e-9 threshold!`);
}

console.log(`✓ PARITY PASSED: max |difference| < 1e-9 (${maxDiff.toExponential(3)}).`);
console.log('✓ Verified: Browser pure-JS tree scorer exactly reproduces Python LightGBM booster.');
