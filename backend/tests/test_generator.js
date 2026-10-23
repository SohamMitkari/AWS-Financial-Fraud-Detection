/**
 * Test script for data generator extracted directly from frontend/index.html
 * Validates realistic fraud prevalence between 0.8% and 2.0% on 50,000 records.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const htmlPath = path.resolve(__dirname, '../../frontend/index.html');
const htmlContent = fs.readFileSync(htmlPath, 'utf-8');

// Extract the script block
const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>[\s\S]*?$/);
if (!scriptMatch) {
  throw new Error('Failed to find <script> block in frontend/index.html');
}

const scriptCode = scriptMatch[1];

// Extract seededRandom and generate functions
const generatorMatch = scriptCode.match(/(function seededRandom[\s\S]*?return out;\s*\})/);
if (!generatorMatch) {
  throw new Error('Failed to extract generate() function from frontend/index.html');
}

const generatorSource = generatorMatch[1];
const sandbox = {};
vm.runInNewContext(generatorSource, sandbox);

if (typeof sandbox.generate !== 'function') {
  throw new Error('Extracted generator is not a function');
}

console.log('Testing REAL generator extracted from frontend/index.html on 50,000 records...');
const n = 50000;
const seed = 20261006;
const records = sandbox.generate(n, seed);

if (records.length !== n) {
  throw new Error(`Expected ${n} records, got ${records.length}`);
}

const fraudCount = records.filter(r => r.isFraud === 1).length;
const fraudRatePct = (fraudCount / n) * 100;

console.log(`Generated: ${n.toLocaleString()} records.`);
console.log(`Fraud count: ${fraudCount.toLocaleString()} (${fraudRatePct.toFixed(3)}%)`);

// Assert realistic class distribution between 0.8% and 2.0%
if (fraudRatePct < 0.8 || fraudRatePct > 2.0) {
  throw new Error(`Fraud rate ${fraudRatePct.toFixed(2)}% is outside expected range [0.8%, 2.0%]!`);
}

console.log('✓ Generator verification passed: realistic fraud rate strictly within [0.8%, 2.0%].');
