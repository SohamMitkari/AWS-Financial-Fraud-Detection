/**
 * Helper to extract and load frontend functions from frontend/index.html
 * using brace-matching rather than fragile regex blocks.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function extractConstant(code, name) {
  const re = new RegExp('\\bconst\\s+' + name + '\\s*=\\s*(\\[[\\s\\S]*?\\]|\\{[\\s\\S]*?\\}|[^;]+);');
  const m = re.exec(code);
  return m ? m[0] : null;
}

function extractFunction(code, fnName) {
  const re = new RegExp('\\b(?:async\\s+)?function\\s+' + fnName + '\\s*\\(', 'g');
  const match = re.exec(code);
  if (!match) return null;
  const startPos = match.index;
  const braceStart = code.indexOf('{', startPos);
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
        i++; // skip escaped char
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

    if (ch === '{') {
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0) {
        return code.slice(startPos, i + 1);
      }
    }
  }
  return null;
}

// Known internal helper dependencies between frontend functions
const DEPENDENCIES = {
  extractCausalFeatures: ['FEATURE_NAMES', 'expectedBalances', 'buildFeatureVector'],
  buildFeatureVector: ['FEATURE_NAMES', 'expectedBalances'],
  generate: ['seededRandom', 'createGenerator'],
  createGenerator: ['seededRandom'],
  scoreRow: ['flattenTree', 'sigmoidJS'],
  prepareModelTrees: ['flattenTree']
};

function loadFrontendFunctions(names, options = {}) {
  const htmlPath = options.htmlPath || path.resolve(__dirname, '../../frontend/index.html');
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');
  const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>[\s\S]*?$/);
  if (!scriptMatch) throw new Error('No <script> block found in frontend/index.html');
  const scriptCode = scriptMatch[1];

  const toLoad = new Set();
  function addWithDeps(name) {
    if (toLoad.has(name)) return;
    const deps = DEPENDENCIES[name] || [];
    for (const d of deps) addWithDeps(d);
    toLoad.add(name);
  }
  for (const n of names) addWithDeps(n);

  const snippets = [];
  for (const item of toLoad) {
    if (item === 'FEATURE_NAMES') {
      const c = extractConstant(scriptCode, 'FEATURE_NAMES');
      if (c) snippets.push(c.replace(/^\bconst\s+/, 'var '));
      else throw new Error('Could not extract constant ' + item);
    } else {
      const fnCode = extractFunction(scriptCode, item);
      if (fnCode) snippets.push(fnCode);
      else throw new Error('Could not extract function ' + item);
    }
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
    Date
  };
  const context = vm.createContext(sandbox);
  const combined = snippets.join('\n\n');
  vm.runInContext(combined, context);

  const result = {};
  for (const item of toLoad) {
    if (sandbox[item] !== undefined) {
      result[item] = sandbox[item];
    }
  }
  return result;
}

module.exports = {
  loadFrontendFunctions,
  extractFunction,
  extractConstant
};
