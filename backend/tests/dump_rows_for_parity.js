/**
 * Dump 2,000 generated transactions and their pure-JS causal feature vectors
 * to a temporary JSON file for cross-language parity testing against Python.
 *
 * Usage:
 *   node backend/tests/dump_rows_for_parity.js [outputPath] [seed]
 */

const fs = require('fs');
const path = require('path');
const { loadFrontendFunctions } = require('./_extract');

const defaultOutputPath = path.resolve(__dirname, 'temp_feature_parity.json');
const outputPath = process.argv[2] ? path.resolve(process.argv[2]) : defaultOutputPath;
const seed = process.argv[3] ? parseInt(process.argv[3], 10) : 20261006;

const { generate, extractCausalFeatures, FEATURE_NAMES } = loadFrontendFunctions([
  'generate',
  'extractCausalFeatures',
  'FEATURE_NAMES'
]);

console.log(`Generating 2,000 rows (seed: ${seed})...`);
const rows = generate(2000, seed);

console.log('Extracting causal feature vectors in JavaScript...');
const { X } = extractCausalFeatures(rows);

const payload = {
  seed,
  feature_names: FEATURE_NAMES,
  rows,
  js_vectors: X
};

fs.writeFileSync(outputPath, JSON.stringify(payload));
console.log(`Successfully dumped ${rows.length} rows and JS feature vectors to ${outputPath}`);
