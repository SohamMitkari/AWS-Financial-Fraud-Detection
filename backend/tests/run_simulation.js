/**
 * End-to-End Simulation Script
 * Generates 50,000 rows, extracts causal features, splits 70/30, and exports
 * a sample payload for Lambda handler local verification.
 */

const fs = require('fs');

function seededRandom(seed) {
  let s = seed >>> 0;
  return function() {
    s += 0x6D2B79F5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateTransactions(n = 50000, seed = 20261006) {
  const rand = seededRandom(seed);
  const types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'];
  const ORIG_POOL_SIZE = 8000;
  const DEST_POOL_SIZE = 3000;

  function pickOrig() {
    const idx = Math.floor(Math.pow(rand(), 2.8) * ORIG_POOL_SIZE);
    return 'C' + (10000000 + idx);
  }
  function pickDest() {
    const idx = Math.floor(Math.pow(rand(), 2.5) * DEST_POOL_SIZE);
    return 'M' + (20000000 + idx);
  }

  const origBalances = new Map();
  const origHistory = new Map();
  const destHistory = new Map();

  const records = [];

  for (let i = 0; i < n; i++) {
    const step = Math.floor(i / 24) + 1;
    const hour = step % 24;
    const type = types[Math.floor(rand() * types.length)];
    const orig = pickOrig();
    const dest = pickDest();

    let oldO = origBalances.get(orig);
    if (oldO === undefined) {
      oldO = Math.round((rand() * 120000 + 100) * 100) / 100;
    }

    let amount;
    if (type === 'PAYMENT' || type === 'DEBIT') {
      amount = Math.round((rand() * 3000 + 5) * 100) / 100;
    } else if (type === 'CASH_IN') {
      amount = Math.round((rand() * 40000 + 50) * 100) / 100;
    } else {
      amount = Math.round((rand() * 150000 + 50) * 100) / 100;
    }

    let newO = Math.max(0, Math.round((oldO - (type === 'CASH_IN' ? -amount : amount)) * 100) / 100);
    const drained = (oldO > 0 && newO === 0 && (type === 'TRANSFER' || type === 'CASH_OUT'));

    const oldD = Math.round((rand() * 180000) * 100) / 100;
    const newD = Math.round((oldD + (type === 'CASH_IN' ? -amount : amount)) * 100) / 100;

    const oHist = origHistory.get(orig) || { count: 0, total: 0 };
    const priorMean = oHist.count > 0 ? (oHist.total / oHist.count) : amount;
    const dHist = destHistory.get(dest) || { count: 0 };
    const isNewDest = dHist.count === 0;

    let riskLogit = -6.3;

    if (type === 'TRANSFER') riskLogit += 1.8;
    else if (type === 'CASH_OUT') riskLogit += 1.5;
    else if (type === 'PAYMENT') riskLogit -= 1.8;
    else if (type === 'DEBIT') riskLogit -= 2.0;
    else if (type === 'CASH_IN') riskLogit -= 2.5;

    if (drained) riskLogit += 1.6;
    if (amount > 75000) riskLogit += 0.9;
    if (amount > 180000) riskLogit += 1.3;
    if (priorMean > 0 && amount > priorMean * 2.5) riskLogit += 1.0;
    if (hour >= 1 && hour <= 5) riskLogit += 0.7;
    if (isNewDest && (type === 'TRANSFER' || type === 'CASH_OUT')) riskLogit += 0.6;

    const noise = (rand() + rand() + rand() - 1.5) * 1.8;
    const totalZ = riskLogit + noise;
    const prob = 1.0 / (1.0 + Math.exp(-totalZ));

    let isFraud = prob > 0.5 ? 1 : 0;
    if (rand() < 0.001) isFraud = 1;
    if (isFraud === 1 && (type === 'PAYMENT' || type === 'DEBIT') && rand() < 0.8) isFraud = 0;

    let finalOldO = oldO;
    let finalNewO = newO;
    let finalOldD = oldD;
    let finalNewD = newD;

    if (rand() < 0.008) {
      finalNewO = Math.round((oldO - amount * 0.45) * 100) / 100;
    }
    if (rand() < 0.004) {
      finalNewD = oldD;
    }

    records.push({
      step,
      type,
      amount,
      nameOrig: orig,
      oldbalanceOrg: finalOldO,
      newbalanceOrig: finalNewO,
      nameDest: dest,
      oldbalanceDest: finalOldD,
      newbalanceDest: finalNewD,
      isFraud
    });

    origBalances.set(orig, finalNewO);
    origHistory.set(orig, { count: oHist.count + 1, total: oHist.total + amount });
    destHistory.set(dest, { count: dHist.count + 1 });
  }

  // Data flaws for Stage 2
  const flawCount = Math.max(4, Math.floor(n * 0.005));
  for (let k = 0; k < flawCount; k++) {
    const idx = Math.floor(rand() * n);
    const r = records[idx];
    if (k % 3 === 0) r.amount = -Math.abs(r.amount);
    else if (k % 3 === 1) r.oldbalanceOrg = null;
    else {
      const targetIdx = Math.floor(rand() * n);
      records[idx] = { ...records[targetIdx] };
    }
  }

  return records;
}

function cleanDataset(records) {
  const seen = new Set();
  const cleaned = [];
  let nullCount = 0;
  let negCount = 0;
  let dupCount = 0;
  let balanceErrors = 0;

  for (const r of records) {
    let hasNull = false;
    let hasNeg = false;

    if (r.amount == null || !Number.isFinite(+r.amount) || r.oldbalanceOrg == null || r.newbalanceOrig == null || r.oldbalanceDest == null || r.newbalanceDest == null) {
      nullCount++;
      hasNull = true;
    }

    const amt = +r.amount || 0;
    if (amt < 0 || +r.oldbalanceOrg < 0 || +r.newbalanceOrig < 0 || +r.oldbalanceDest < 0 || +r.newbalanceDest < 0) {
      negCount++;
      hasNeg = true;
    }

    const oldO = Math.max(0, +r.oldbalanceOrg || 0);
    const newO = Math.max(0, +r.newbalanceOrig || 0);
    const oldD = Math.max(0, +r.oldbalanceDest || 0);
    const newD = Math.max(0, +r.newbalanceDest || 0);
    const safeAmt = Math.abs(amt);

    const origErr = Math.abs((oldO - safeAmt) - newO) > 0.05;
    const destErr = Math.abs((oldD + safeAmt) - newD) > 0.05;
    if (origErr || destErr) {
      balanceErrors++;
    }

    const key = [r.step, r.type, r.nameOrig, r.nameDest, safeAmt].join('|');
    if (seen.has(key)) {
      dupCount++;
      continue;
    }
    seen.add(key);

    cleaned.push({
      step: +r.step || 1,
      type: r.type,
      amount: safeAmt,
      nameOrig: r.nameOrig,
      oldbalanceOrg: oldO,
      newbalanceOrig: newO,
      nameDest: r.nameDest,
      oldbalanceDest: oldD,
      newbalanceDest: newD,
      isFraud: +r.isFraud === 1 ? 1 : 0
    });
  }

  return { cleaned, auditCounts: { nullCount, negCount, dupCount, balanceErrors } };
}

function computeFeatures(records) {
  const origHistory = new Map();
  const destHistory = new Map();

  const FEATURE_NAMES = [
    'amount_log1p',
    'balance_orig_diff',
    'balance_dest_diff',
    'orig_balance_error',
    'dest_balance_error',
    'orig_drained',
    'hour_of_day',
    'type_TRANSFER',
    'type_CASH_OUT',
    'type_PAYMENT',
    'type_DEBIT',
    'type_CASH_IN',
    'orig_prior_count',
    'orig_prior_mean_amount',
    'amount_vs_orig_mean',
    'dest_prior_count',
    'dest_prior_max_amount',
    'is_new_dest',
    'orig_txn_last_24_steps'
  ];

  const X = [];
  const y = [];

  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    const amount = +r.amount || 0;
    const oldO = +r.oldbalanceOrg || 0;
    const newO = +r.newbalanceOrig || 0;
    const oldD = +r.oldbalanceDest || 0;
    const newD = +r.newbalanceDest || 0;
    const step = +r.step || 1;
    const type = String(r.type || '').toUpperCase();
    const orig = String(r.nameOrig || '');
    const dest = String(r.nameDest || '');

    const amount_log1p = Math.log1p(Math.max(0, amount));
    const balance_orig_diff = oldO - newO;
    const balance_dest_diff = newD - oldD;
    const orig_balance_error = Math.abs((oldO - amount) - newO) > 0.05 ? 1 : 0;
    const dest_balance_error = Math.abs((oldD + amount) - newD) > 0.05 ? 1 : 0;
    const orig_drained = (oldO > 0 && newO === 0) ? 1 : 0;
    const hour_of_day = step % 24;

    const type_TRANSFER = type === 'TRANSFER' ? 1 : 0;
    const type_CASH_OUT = type === 'CASH_OUT' ? 1 : 0;
    const type_PAYMENT = type === 'PAYMENT' ? 1 : 0;
    const type_DEBIT = type === 'DEBIT' ? 1 : 0;
    const type_CASH_IN = type === 'CASH_IN' ? 1 : 0;

    const origState = origHistory.get(orig) || { count: 0, totalAmount: 0, lastSteps: [] };
    const orig_prior_count = origState.count;
    const orig_prior_mean_amount = origState.count > 0 ? (origState.totalAmount / origState.count) : 0;
    const amount_vs_orig_mean = orig_prior_mean_amount > 0 ? (amount / orig_prior_mean_amount) : 1.0;

    const destState = destHistory.get(dest) || { count: 0, maxAmount: 0 };
    const dest_prior_count = destState.count;
    const dest_prior_max_amount = destState.maxAmount;
    const is_new_dest = dest_prior_count === 0 ? 1 : 0;

    const recentTxns = origState.lastSteps.filter(s => s > step - 24);
    const orig_txn_last_24_steps = recentTxns.length;

    const vec = [
      +amount_log1p.toFixed(4),
      +balance_orig_diff.toFixed(4),
      +balance_dest_diff.toFixed(4),
      orig_balance_error,
      dest_balance_error,
      orig_drained,
      hour_of_day,
      type_TRANSFER,
      type_CASH_OUT,
      type_PAYMENT,
      type_DEBIT,
      type_CASH_IN,
      orig_prior_count,
      +orig_prior_mean_amount.toFixed(4),
      +amount_vs_orig_mean.toFixed(4),
      dest_prior_count,
      +dest_prior_max_amount.toFixed(4),
      is_new_dest,
      orig_txn_last_24_steps
    ];

    X.push(vec);
    y.push(+r.isFraud === 1 ? 1 : 0);

    origHistory.set(orig, {
      count: origState.count + 1,
      totalAmount: origState.totalAmount + amount,
      lastSteps: [...origState.lastSteps.filter(s => s > step - 48), step]
    });

    destHistory.set(dest, {
      count: destState.count + 1,
      maxAmount: Math.max(destState.maxAmount, amount)
    });
  }

  return { X, y, featureNames: FEATURE_NAMES };
}

// Generate full pipeline
const raw = generateTransactions(50000, 20261006);
const { cleaned, auditCounts } = cleanDataset(raw);
console.log('Stage 2 Cleaned Audit Counts:', auditCounts);

const { X, y, featureNames } = computeFeatures(cleaned);
console.log(`Computed ${X.length} causal feature vectors with ${featureNames.length} features.`);

// Split 70/30
const trainCount = Math.floor(X.length * 0.7);
const X_train_full = X.slice(0, trainCount);
const y_train_full = y.slice(0, trainCount);
const X_test = X.slice(trainCount);
const y_test = y.slice(trainCount);

// Undersample legitimate train class if payload exceeds 12,000 rows
let X_train = [];
let y_train = [];
const MAX_TRAIN_ROWS = 12000;

if (X_train_full.length > MAX_TRAIN_ROWS) {
  const fraudIndices = [];
  const legitIndices = [];
  for (let i = 0; i < y_train_full.length; i++) {
    if (y_train_full[i] === 1) fraudIndices.push(i);
    else legitIndices.push(i);
  }
  const allowedLegit = MAX_TRAIN_ROWS - fraudIndices.length;
  // Step through legit indices evenly
  const step = legitIndices.length / allowedLegit;
  const sampledLegit = [];
  for (let k = 0; k < allowedLegit; k++) {
    sampledLegit.push(legitIndices[Math.floor(k * step)]);
  }
  const selected = [...fraudIndices, ...sampledLegit].sort((a, b) => a - b);
  X_train = selected.map(idx => X_train_full[idx]);
  y_train = selected.map(idx => y_train_full[idx]);
  console.log(`Train partition undersampled: kept ${selected.length} of ${X_train_full.length} rows (${fraudIndices.length} frauds). Test set untouched: ${X_test.length} rows.`);
} else {
  X_train = X_train_full;
  y_train = y_train_full;
}

const payload = {
  feature_names: featureNames,
  train: { X: X_train, y: y_train },
  test: { X: X_test, y: y_test },
  options: { epochs: 15, learning_rate: 0.015, l2: 1e-4, seed: 42 },
  meta: { split: '70/30', n_raw: raw.length, n_clean: cleaned.length }
};

fs.writeFileSync('backend/tests/sample_payload.json', JSON.stringify(payload));
console.log('Exported backend/tests/sample_payload.json successfully.');
