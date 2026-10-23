/**
 * Headless browser test for Real Dataset Mode (Stages 1 to 7 and Dashboard).
 * Uses puppeteer-core to launch local Edge/Chrome,
 * mocks cloud backend /train response,
 * loads tests/fixtures/real/creditcard_sample300.csv,
 * CLICKS the real UI buttons through Stages 1 to 6,
 * then clicks Stage 7 and Dashboard tab, asserting they render with zero console errors.
 */

const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');
const http = require('http');
const assert = require('assert');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_DIR = path.join(ROOT_DIR, 'frontend');
const FIXTURE_PATH = path.join(ROOT_DIR, 'tests/fixtures/real/ethereum_sample300.csv');

function findBrowser() {
  if (process.env.BROWSER_PATH && fs.existsSync(process.env.BROWSER_PATH)) {
    return process.env.BROWSER_PATH;
  }
  const platform = process.platform;
  const candidates = [];

  if (platform === 'win32') {
    const prefixes = [
      process.env['PROGRAMFILES(X86)'],
      process.env.PROGRAMFILES,
      process.env.LOCALAPPDATA,
      'C:\\Program Files (x86)',
      'C:\\Program Files'
    ].filter(Boolean);

    for (const prefix of prefixes) {
      candidates.push(path.join(prefix, 'Microsoft/Edge/Application/msedge.exe'));
      candidates.push(path.join(prefix, 'Google/Chrome/Application/chrome.exe'));
    }
  } else if (platform === 'darwin') {
    candidates.push(
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      path.join(process.env.HOME || '', 'Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'),
      path.join(process.env.HOME || '', 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    );
  } else {
    candidates.push(
      '/usr/bin/microsoft-edge',
      '/usr/bin/microsoft-edge-stable',
      '/usr/bin/google-chrome',
      '/usr/bin/google-chrome-stable',
      '/usr/bin/chromium',
      '/usr/bin/chromium-browser',
      '/snap/bin/chromium'
    );
  }

  for (const p of candidates) {
    if (p && fs.existsSync(p)) return p;
  }

  try {
    const cmd = platform === 'win32'
      ? 'where msedge 2>nul || where chrome 2>nul'
      : 'which google-chrome || which microsoft-edge || which chromium';
    const out = require('child_process').execSync(cmd, { stdio: ['pipe', 'pipe', 'ignore'], encoding: 'utf-8' }).trim().split(/\r?\n/)[0];
    if (out && fs.existsSync(out)) return out;
  } catch (_) {}

  throw new Error('No compatible browser (Edge or Chrome) found. Please set BROWSER_PATH environment variable.');
}

function createStaticServer(port = 8086) {
  const server = http.createServer((req, res) => {
    let reqPath = req.url.split('?')[0];
    if (reqPath === '/' || reqPath === '/index.html') reqPath = '/index.html';
    if (reqPath === '/config.js') {
      res.writeHead(200, {
        'Content-Type': 'application/javascript',
        'Access-Control-Allow-Origin': '*'
      });
      res.end(`window.APP_CONFIG = { API_BASE: "http://127.0.0.1:${port}/api" };`);
      return;
    }
    const filePath = path.join(FRONTEND_DIR, reqPath.replace(/^\//, ''));
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      const ct = ext === '.html' ? 'text/html' : ext === '.js' ? 'application/javascript' : ext === '.css' ? 'text/css' : 'image/jpeg';
      res.writeHead(200, {
        'Content-Type': ct,
        'Access-Control-Allow-Origin': '*'
      });
      fs.createReadStream(filePath).pipe(res);
    } else {
      res.writeHead(404);
      res.end('Not Found');
    }
  });

  return new Promise((resolve, reject) => {
    server.listen(port, () => resolve(server));
    server.on('error', reject);
  });
}

async function runRealBrowserVerification() {
  const port = 8086;
  console.log(`Starting mock static HTTP server on port ${port}...`);
  const server = await createStaticServer(port);

  const browserPath = findBrowser();
  console.log(`Launching headless browser: ${browserPath}...`);
  const browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1280,900']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });

    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Ignore expected network 404 for optional favicon if any
        if (text.includes('favicon.ico')) return;
        consoleErrors.push(text);
        console.error('   [Browser Console Error]:', text);
      }
    });

    page.on('pageerror', err => {
      consoleErrors.push(err.toString());
      console.error('   [Browser Page Error]:', err.toString());
    });

    // Mock cloud /train and /predict endpoints
    await page.setRequestInterception(true);
    page.on('request', req => {
      const url = req.url();
      if (url.includes('/train')) {
        const mockModelResponse = {
          status: 'SUCCESS',
          model_version: 'lgb-real-v1-mock',
          model: {
            trees: [
              {
                split_feature: 0,
                threshold: 0.0,
                decision_type: '<=',
                default_left: true,
                missing_type: 'None',
                left_child: { leaf_value: -1.24 },
                right_child: { leaf_value: 2.15 }
              },
              {
                split_feature: 1,
                threshold: 0.5,
                decision_type: '<=',
                default_left: true,
                missing_type: 'None',
                left_child: { leaf_value: -0.85 },
                right_child: { leaf_value: 0.92 }
              }
            ],
            feature_names: ['V1', 'V2', 'V3'],
            thresholds: { decision: 0.50, review: 0.35, block: 0.75 },
            model_version: 'lgb-real-v1-mock',
            objective: 'binary'
          },
          metrics: {
            precision: 0.85,
            recall: 0.80,
            f1: 0.82,
            accuracy: 0.99,
            roc_auc: 0.95,
            pr_auc: 0.85
          },
          confusion: { tp: 24, fp: 4, fn: 6, tn: 266 },
          baseline: { accuracy: 0.90 },
          importance: [
            { feature: 'V1', gain_pct: 50.0 },
            { feature: 'V2', gain_pct: 30.0 },
            { feature: 'V3', gain_pct: 20.0 }
          ],
          trees_count: 2,
          best_iteration: 2,
          params: { learning_rate: 0.05, num_leaves: 15, max_depth: 6, lambda_l2: 1.0, scale_pos_weight: 1.0 },
          timings: { train_ms: 45.2, total_ms: 70.1 }
        };

        req.respond({
          status: 200,
          contentType: 'application/json',
          headers: { 'Access-Control-Allow-Origin': '*' },
          body: JSON.stringify(mockModelResponse)
        });
        return;
      }

      req.continue();
    });

    console.log(`Navigating to http://127.0.0.1:${port}/index.html...`);
    await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'networkidle0' });

    // 1. Switch to Real Dataset mode by clicking #modeReal button
    console.log('Clicking #modeReal to enter Real Dataset mode...');
    await page.waitForSelector('#modeReal');
    await page.click('#modeReal');

    // 2. Upload creditcard_sample300.csv
    console.log(`Uploading fixture: ${FIXTURE_PATH}...`);
    await page.waitForSelector('#realCsvFile');
    const fileInput = await page.$('#realCsvFile');
    await fileInput.uploadFile(FIXTURE_PATH);

    // Wait for parse and ingestion completion
    await page.waitForSelector('#realMsg .notice.success', { timeout: 10000 });
    console.log('   ✓ Ingested CSV and auto-detected columns.');

    // 3. Click #run for Stage 1 (Execute Stage 1)
    console.log('Clicking #run for Stage 1...');
    await page.click('#run');
    await page.waitForFunction(() => document.getElementById('status')?.textContent.includes('Stage 1 executed ✓'), { timeout: 5000 });
    console.log('   ✓ Stage 1 completed.');

    // Helper to click stage nav button and click #run
    async function runStageByClick(stageIndex, stageName) {
      console.log(`Navigating to Stage ${stageIndex + 1} (${stageName}) via UI click...`);
      await page.evaluate((idx) => {
        const buttons = document.querySelectorAll('#nav button');
        buttons[idx].click();
      }, stageIndex);

      await page.waitForSelector('#run');
      console.log(`Clicking #run for Stage ${stageIndex + 1}...`);
      await page.click('#run');
      await page.waitForFunction(
        (idx) => document.getElementById('status')?.textContent.includes(`Stage ${idx + 1} executed ✓`),
        { timeout: 5000 },
        stageIndex
      );
      console.log(`   ✓ Stage ${stageIndex + 1} completed.`);
    }

    // 4. Stages 2 to 5 UI clicks
    await runStageByClick(1, 'Cleaning');
    await runStageByClick(2, 'EDA');
    await runStageByClick(3, 'Feature Engineering');
    await runStageByClick(4, 'Splitting');

    // 5. Stage 6: Click Stage 6 nav, enter training key, and click #run
    console.log('Navigating to Stage 6 (Model Training) via UI click...');
    await page.evaluate(() => {
      document.querySelectorAll('#nav button')[5].click();
    });

    await page.waitForSelector('#trainApiKeyInput');
    console.log('Typing Training key into #trainApiKeyInput...');
    await page.type('#trainApiKeyInput', 'mock-train-key-12345');

    console.log('Clicking #run for Stage 6 (Train & Evaluate)...');
    await page.click('#run');

    await page.waitForFunction(() => document.getElementById('status')?.textContent.includes('Stage 6 executed ✓'), { timeout: 8000 });
    console.log('   ✓ Stage 6 completed (LightGBM trained & test slice scored in pure JS).');

    // 6. Stage 7: Click Stage 7 nav and click #run (Operational Scoring Replay)
    console.log('Navigating to Stage 7 (Operational Scoring) via UI click...');
    await page.evaluate(() => {
      document.querySelectorAll('#nav button')[6].click();
    });

    await page.waitForSelector('#run');
    console.log('Clicking #run for Stage 7...');
    await page.click('#run');

    await page.waitForFunction(() => document.getElementById('status')?.textContent.includes('Stage 7 executed ✓'), { timeout: 5000 });
    console.log('   ✓ Stage 7 executed.');

    // Assert Stage 7 rendered components
    const stage7Rendered = await page.evaluate(() => {
      const liveTables = document.querySelectorAll('.live-table tbody');
      const breakdownRows = liveTables[0] ? liveTables[0].querySelectorAll('tr').length : 0;
      const top50Rows = liveTables[1] ? liveTables[1].querySelectorAll('tr').length : 0;
      const bodyText = document.body.innerText;
      return {
        breakdownRows,
        top50Rows,
        hasHeldOutNotice: bodyText.includes('replays held-out test rows') && bodyText.includes('not a live stream'),
        hasBreakdownTable: bodyText.includes('Operational Triage Breakdown'),
        hasBlockKpi: bodyText.includes('Block Action Count'),
        hasReviewKpi: bodyText.includes('Review Action Count'),
        hasAllowKpi: bodyText.includes('Allow Action Count')
      };
    });

    assert.strictEqual(stage7Rendered.breakdownRows, 3, `Stage 7 breakdown table must contain 3 rows (Allow, Review, Block), got ${stage7Rendered.breakdownRows}`);
    assert.strictEqual(stage7Rendered.top50Rows, 50, `Stage 7 top-50 table must contain 50 rows (got ${stage7Rendered.top50Rows})`);
    assert.strictEqual(stage7Rendered.hasHeldOutNotice, true, 'Stage 7 must display the held-out replay notice');
    assert.strictEqual(stage7Rendered.hasBreakdownTable, true, 'Stage 7 must display the triage breakdown table');
    assert.strictEqual(stage7Rendered.hasBlockKpi, true, 'Stage 7 must display Block KPI');
    assert.strictEqual(stage7Rendered.hasReviewKpi, true, 'Stage 7 must display Review KPI');
    assert.strictEqual(stage7Rendered.hasAllowKpi, true, 'Stage 7 must display Allow KPI');
    console.log(`   ✓ Stage 7 rendered: 50 rows in top-50 table, 3 breakdown rows, held-out replay notice, and action triage stats.`);

    // 7. Dashboard Tab: Click Dashboard nav and assert components
    console.log('Navigating to Dashboard tab via UI click...');
    await page.evaluate(() => {
      document.querySelectorAll('#nav button')[7].click();
    });

    await page.waitForSelector('.live-table');

    const dashRendered = await page.evaluate(() => {
      const bodyText = document.body.innerText;
      const kpis = Array.from(document.querySelectorAll('.kpi .klabel')).map(el => el.textContent.trim());
      const svgs = document.querySelectorAll('svg.visual-svg').length;
      const rows = document.querySelectorAll('.live-table tbody tr').length;
      return {
        bodyText,
        kpis,
        svgs,
        rows
      };
    });

    assert.strictEqual(dashRendered.kpis.some(k => k.includes('Test Rows')), true, 'Dashboard must show Test Rows KPI');
    assert.strictEqual(dashRendered.kpis.some(k => k.includes('True Frauds')), true, 'Dashboard must show True Frauds KPI');
    assert.strictEqual(dashRendered.kpis.some(k => k.includes('PR-AUC')), true, 'Dashboard must show PR-AUC KPI');
    assert.strictEqual(dashRendered.kpis.some(k => k.includes('Recall')), true, 'Dashboard must show Recall KPI');
    assert.strictEqual(dashRendered.kpis.some(k => k.includes('Precision')), true, 'Dashboard must show Precision KPI');
    assert.strictEqual(dashRendered.svgs >= 3, true, `Dashboard must render at least 3 SVG charts (got ${dashRendered.svgs})`);
    assert.strictEqual(dashRendered.rows > 0, true, 'Dashboard replay table must render rows');
    console.log(`   ✓ Dashboard rendered: 5 KPI cards (Test Rows, Frauds, PR-AUC, Recall, Precision), 3 visual SVG charts, and ${dashRendered.rows} replayed rows.`);

    // Test Dashboard Filter Click
    console.log('Clicking Dashboard action filter button [data-real-dash-filter="Block"]...');
    await page.click('[data-real-dash-filter="Block"]');
    await page.waitForFunction(() => document.body.innerText.includes('Filter: Block'), { timeout: 3000 });
    console.log('   ✓ Dashboard filter applied successfully.');

    // 8. Live Replay Stream Panel assertions
    console.log('Testing Live Replay Stream panel on Dashboard...');
    await page.waitForSelector('#realStreamStart');
    await page.waitForSelector('#realStreamPause');
    await page.waitForSelector('#realStreamReset');
    await page.waitForSelector('#realStreamSpeed');
    await page.waitForSelector('#realStreamBody');

    const initialStreamRows = await page.evaluate(() => document.querySelectorAll('#realStreamBody tr').length);
    assert.strictEqual(initialStreamRows, 0, 'Stream table should start with 0 rows');

    // Click Start
    console.log('Clicking #realStreamStart to start stream replay...');
    await page.click('#realStreamStart');

    // Assert: rows appear incrementally (count rises over time, not all at once)
    console.log('Verifying rows appear incrementally over time...');
    await new Promise(r => setTimeout(r, 250));
    const count1 = await page.evaluate(() => document.querySelectorAll('#realStreamBody tr').length);
    assert.strictEqual(count1 >= 1, true, `Expected at least 1 row to appear shortly after Start, got ${count1}`);

    await new Promise(r => setTimeout(r, 500));
    const count2 = await page.evaluate(() => document.querySelectorAll('#realStreamBody tr').length);
    assert.strictEqual(count2 > count1, true, `Rows must appear incrementally (count1=${count1}, count2=${count2})`);
    console.log(`   ✓ Incremental row appearance verified (${count1} -> ${count2} rows).`);

    // Assert: at least one toast appears for a Block row
    console.log('Verifying Block notification toast...');
    const toastCount = await page.evaluate(() => document.querySelectorAll('#realToastContainer .real-toast').length);
    assert.strictEqual(toastCount >= 1, true, `Expected at least one Block toast, got ${toastCount}`);
    const toastText = await page.evaluate(() => document.querySelector('#realToastContainer .real-toast')?.innerText || '');
    assert.strictEqual(toastText.includes('Row #') && toastText.includes('Score'), true, `Toast must show row number and score (got: ${toastText})`);
    console.log(`   ✓ Block toast confirmed: "${toastText.replace(/\s+/g, ' ')}"`);

    // Click Pause
    console.log('Clicking #realStreamPause to pause replay...');
    await page.click('#realStreamPause');
    const pausedRowCount = await page.evaluate(() => document.querySelectorAll('#realStreamBody tr').length);

    // Wait 3.5 seconds to assert toast removal and that Pause stopped new rows
    console.log('Waiting 3.5 seconds to assert toast is removed from DOM and rows remain paused...');
    await new Promise(r => setTimeout(r, 3500));

    const toastsAfterWait = await page.evaluate(() => document.querySelectorAll('#realToastContainer .real-toast').length);
    assert.strictEqual(toastsAfterWait, 0, `Toast must be removed from the DOM after about 3 seconds (waited 3.5s, remaining: ${toastsAfterWait})`);
    console.log('   ✓ Block toast successfully removed from DOM after 3.5s.');

    const rowCountAfterPause = await page.evaluate(() => document.querySelectorAll('#realStreamBody tr').length);
    assert.strictEqual(rowCountAfterPause, pausedRowCount, `Pause must stop new rows from being added (${pausedRowCount} vs ${rowCountAfterPause})`);
    console.log(`   ✓ Pause confirmed: row count remained stable at ${pausedRowCount}.`);

    // Assert: Reset clears the table and counters
    console.log('Clicking #realStreamReset to clear replay...');
    await page.click('#realStreamReset');
    const resetState = await page.evaluate(() => ({
      rows: document.querySelectorAll('#realStreamBody tr').length,
      replayed: document.getElementById('realCounterReplayed')?.textContent?.trim(),
      allow: document.getElementById('realCounterAllow')?.textContent?.trim(),
      review: document.getElementById('realCounterReview')?.textContent?.trim(),
      block: document.getElementById('realCounterBlock')?.textContent?.trim(),
      frauds: document.getElementById('realCounterFraudsCaught')?.textContent?.trim()
    }));

    assert.strictEqual(resetState.rows, 0, `Reset must clear table rows (got ${resetState.rows})`);
    assert.strictEqual(resetState.replayed, '0', `Reset must clear replayed counter to 0 (got ${resetState.replayed})`);
    assert.strictEqual(resetState.allow, '0', `Reset must clear allow counter to 0 (got ${resetState.allow})`);
    assert.strictEqual(resetState.review, '0', `Reset must clear review counter to 0 (got ${resetState.review})`);
    assert.strictEqual(resetState.block, '0', `Reset must clear block counter to 0 (got ${resetState.block})`);
    assert.strictEqual(resetState.frauds, '0', `Reset must clear frauds caught counter to 0 (got ${resetState.frauds})`);
    console.log('   ✓ Reset confirmed: table and all counters cleared to 0.');

    // Assert zero console errors
    assert.strictEqual(consoleErrors.length, 0, `Expected 0 console errors, but got ${consoleErrors.length}: ${JSON.stringify(consoleErrors)}`);
    console.log('   ✓ Verified 0 console errors across all stages and dashboard!');

    console.log('\n=== REAL DATASET BROWSER VERIFICATION PASSED ===\n');
  } finally {
    await browser.close();
    server.close();
  }
}

runRealBrowserVerification().catch(err => {
  console.error('\n❌ Real dataset browser verification failed:', err);
  process.exit(1);
});
