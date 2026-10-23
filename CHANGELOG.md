# CHANGELOG

## Baseline Test Results & Benchmark

### Test Commands Status
- `python -m pytest backend/tests/test_model.py -q`: **PASS** (15 passed in 7.22s)
- `node backend/tests/test_generator.js`: **PASS** (50,000 records, 662 frauds / 1.324%, realistic fraud rate strictly within [0.8%, 2.0%])
- `node backend/tests/test_causal.js`: **PASS** (zero lookahead leakage detected; Rows 0..3 invariant to future mutations; Rows 0..4 invariant to future row appends)
- `python backend/tests/compare_before_after.py`: **PASS** (completed in 205.92 ms for LightGBM training)
- `node backend/tests/verify_parity.js`: **EXPECTED TO FAIL** (Status: FAIL - `Error: Failed to extract scoreRow from index.html`)

### Baseline Benchmark Table (`python backend/tests/compare_before_after.py`)
```
======================================================================================
BENCHMARK COMPARISON ON TEST PARTITION (14,976 records, 174 frauds, prevalence: 1.16%)
======================================================================================
Metric                           | Logistic Regression     | LightGBM (Trees)       
--------------------------------------------------------------------------------------
Model Architecture               | L2 Logistic (Adam)      | Gradient-Boosted Trees 
Data Scaler Required             | YES (StandardScaler)    | NONE (Invariant)       
Number of Trees                  | N/A (Linear weights)    | 104 trees (best 104)   
Decision Threshold (F1 cut)      | 0.81                    | 0.85                   
Precision                        | 12.00                 % | 18.58                 %
Recall (Detection Rate)          | 70.69                 % | 73.56                 %
F1 Score                         | 20.52                 % | 29.66                 %
Accuracy                         | 93.64                 % | 95.95                 %
ROC-AUC                          | 0.9432                  | 0.9472                 
PR-AUC (Headline)                | 0.1639                  | 0.2686                 
Baseline Accuracy                | 98.84                 % | 98.84                 %
Training Latency                 | 1231.69 ms              | 205.92 ms              
Model Artifact Size              | 1.23 KB                 | 275.88 KB              
Confusion Matrix                 | 123 TP / 902 FP / 51 FN / 13900 TN | 128 TP / 561 FP / 46 FN / 14241 TN
======================================================================================
```

---

## Task Log Entries

- What changed: Created FIXLOG.md defining repository rules and recorded baseline execution results and benchmark table.
- Test results: pytest (15 passed in 7.22s), test_generator.js (PASS), test_causal.js (PASS), compare_before_after.py (PASS), verify_parity.js (EXPECTED FAIL).
- Suggested commit: docs: initialize FIXLOG.md with rules and baseline benchmarks

- What changed: Extracted fit_model from handle_train in backend/lambda_function.py and updated backend/tests/generate_parity_data.py to call lf.fit_model.
- Test results: pytest (15 passed in 7.55s), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), test_causal.js (PASS), compare_before_after.py (PASS: LightGBM metrics identical - 104 trees, 0.85 threshold, 18.58% precision, 73.56% recall, 29.66% F1, 0.9472 ROC-AUC, 0.2686 PR-AUC).
- Suggested commit: refactor(backend): extract fit_model from handle_train and reuse in parity data generator

- What changed: Rewrote derive_thresholds in backend/lambda_function.py with 0.01 numpy grid search, block fallback with block_target_met reporting, threshold_diagnostics, warnings, and added unit tests in backend/tests/test_model.py.
- Test results: pytest (19 passed in 7.49s), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), test_causal.js (PASS), compare_before_after.py (PASS: LightGBM metrics identical - 104 trees, 0.85 threshold, 18.58% precision, 73.56% recall, 29.66% F1, 0.9472 ROC-AUC, 0.2686 PR-AUC).
- Suggested commit: feat(backend): rewrite derive_thresholds with numpy grid, diagnostic curves, and fallback warnings

- What changed: Replaced blanket majority-class undersampling in frontend/index.html trainModelStage with a 5,000,000 character payload threshold check and fit-slice-only undersampling that keeps validation and test partitions untouched in chronological order.
- Test results: pytest (19 passed in 7.50s), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), test_causal.js (PASS).
- Suggested commit: feat(frontend): preserve full natural-rate train partition under 5MB and restrict undersampling to fit slice

- What changed: Added expectedBalances and buildFeatureVector helpers in frontend/index.html with 0.05 balance error tolerance, refactored extractCausalFeatures, scoreNewLiveTransaction, and Stage 2 to call them.
- Test results: pytest (19 passed in 7.48s), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), test_causal.js (PASS). orig_balance_error=1 rates on 50k rows: CASH_IN 1.470% (145/9,867), CASH_OUT 0.947% (96/10,132), DEBIT 1.166% (117/10,035), PAYMENT 0.886% (88/9,937), TRANSFER 0.708% (71/10,029), Overall 1.034% (517/50,000).
- Suggested commit: feat(frontend): centralize balance calculation and causal feature vector extraction across pipeline stages

- What changed: Refactored generate into stateful createGenerator(seed) preserving identical output (SHA-256 verified), converted startLive to use a single persistent generator instance with pre-label step computation, removed post-hoc t.step override, tagged scored rows with split ('train' | 'test' | 'live'), isolated dashboard KPI cards to test rows, added separate in-sample training summary, and labeled live stream panel 'Simulated Stream'.
- Test results: generate(1000, 20261006) SHA-256 hash verified identical (f8555b75b238fa4f79068eb531b366aea51458944a393658ff86e0aadef598a3), pytest (19 passed in 5.72s), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), test_causal.js (PASS), compare_before_after.py (PASS: LightGBM metrics identical - 104 trees, 0.85 threshold, 18.58% precision, 73.56% recall, 29.66% F1, 0.9472 ROC-AUC, 0.2686 PR-AUC).
- Suggested commit: feat(frontend): stateful generator, persistent live stream, test-isolated dashboard KPIs, and in-sample metrics

- What changed: Created backend/tests/_extract.js to extract frontend functions via brace-matching, rewrote test_causal.js to verify zero lookahead leakage across 50 cut points on 5,000 rows with real extractCausalFeatures and generate, and fixed verify_parity.js to use _extract.
- Test results: verify_parity.js (PASS: max diff 2.220e-16 < 1e-9), test_causal.js (PASS: 50 cut points invariant under mutate/delete/append), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), pytest (19 passed in 5.22s), compare_before_after.py (PASS: LightGBM metrics identical - 104 trees, 0.85 threshold, 18.58% precision, 73.56% recall, 29.66% F1, 0.9472 ROC-AUC, 0.2686 PR-AUC).
- Suggested commit: test(backend): implement brace-matching frontend function extractor and verify causal invariance and prediction parity

### Benchmark Table After Regenerating `sample_payload.json` (`python backend/tests/compare_before_after.py`)
```
======================================================================================
BENCHMARK COMPARISON ON TEST PARTITION (14,976 records, 174 frauds, prevalence: 1.16%)
======================================================================================
Metric                           | Logistic Regression     | LightGBM (Trees)       
--------------------------------------------------------------------------------------
Model Architecture               | L2 Logistic (Adam)      | Gradient-Boosted Trees 
Data Scaler Required             | YES (StandardScaler)    | NONE (Invariant)       
Number of Trees                  | N/A (Linear weights)    | 105 trees (best 105)   
Decision Threshold (F1 cut)      | 0.87                    | 0.91                   
Precision                        | 18.23                 % | 25.15                 %
Recall (Detection Rate)          | 43.68                 % | 48.28                 %
F1 Score                         | 25.72                 % | 33.07                 %
Accuracy                         | 97.07                 % | 97.73                 %
ROC-AUC                          | 0.9420                  | 0.9430                 
PR-AUC (Headline)                | 0.1641                  | 0.2577                 
Baseline Accuracy                | 98.84                 % | 98.84                 %
Training Latency                 | 2048.28 ms              | 236.68 ms              
Model Artifact Size              | 32.95 KB                | 278.94 KB              
Confusion Matrix                 | 76 TP / 341 FP / 98 FN / 14461 TN | 84 TP / 250 FP / 90 FN / 14552 TN
======================================================================================
```

- What changed: Auto-detected Edge/Chrome across platforms in verify_browser.js, routed screenshots to gitignored ./artifacts/screenshots, created backend/tests/export_payload.js using real generate, Stage 2 cleaning, extractCausalFeatures, 70/30 split, and trainModelStage sampling, regenerated sample_payload.json with seed 20261006, and added test:py, test:js, test:all scripts to package.json.
- Test results: npm run test:all (PASS: pytest 19 passed in 5.28s; test_generator.js PASS: 50,000 records, 662 frauds / 1.324%; test_causal.js PASS: 50 cut points invariant; verify_parity.js PASS: max diff 2.220e-16 < 1e-9), compare_before_after.py (PASS: 105 trees, 0.91 threshold, 25.15% precision, 48.28% recall, 33.07% F1, 0.9430 ROC-AUC, 0.2577 PR-AUC on full 34,942 natural train rows).
- Suggested commit: feat(tests): add cross-platform browser detection, export_payload generator, regenerated natural train payload, and npm test scripts

- What changed: Created backend/features.py porting the 19 causal features and expected_balances from frontend/index.html as a streaming function over dict rows, added backend/tests/dump_rows_for_parity.js to export 2,000 rows with JS vectors to gitignored temp JSON, and added backend/tests/test_feature_parity.py asserting cross-language parity within 1e-6.
- Test results: pytest test_feature_parity.py (4 passed in 0.06s: max diff 0.0 across 2,000 rows), test:py (23 passed in 5.24s), test:js (test_generator.js PASS: 50k rows, 662 frauds / 1.324%; test_causal.js PASS: 50 cut points invariant; verify_parity.js PASS: max diff 2.220e-16 < 1e-9), test:all PASS.
- Suggested commit: feat(backend): port 19 streaming causal features to Python and verify zero-difference cross-language parity

- What changed: Created backend/tests/eval_paysim.py supporting --csv, optional [--max-rows N], and [--drop-post-balance] (dropping features 1 to 5) with missing file instructions, 70/30 chronological split, 80/20 train fit/val split, LightGBM training with fit_model, threshold derivation, evaluation on test set, writing results/paysim.json, and added data/ to .gitignore.
- Test results: eval_paysim.py (100k rows evaluated in 1.44s: 72 trees, PR-AUC 0.0053, ROC-AUC 0.7418, written to results/paysim.json; missing file instructions verified), npm run test:all (23 pytest passed in 5.32s; test_generator.js PASS; test_causal.js PASS; verify_parity.js PASS: max diff 2.220e-16 < 1e-9).
- Suggested commit: feat(tests): implement PaySim benchmark evaluation script with post-balance feature ablation and json reporting

- What changed: Created backend/tests/run_benchmark.py supporting seeds 1 to 5, generating temp payloads with export_payload.js, training Adam logistic regression and LightGBM (fit_model + derive_thresholds), evaluating generative Oracle risk logit baseline, computing test PR-AUC, ROC-AUC, precision, recall, 95% bootstrap CI (1,000 resamples), reporting mean ± std and median training time, and writing results/benchmark.json and results/benchmark.md.
- Test results: run_benchmark.py across 5 seeds (PR-AUC: LR 0.1608 ± 0.0366 [CI: 0.1295, 0.2072], LightGBM 0.2238 ± 0.0244 [CI: 0.1792, 0.2833], Oracle 0.2220 ± 0.0203 [CI: 0.1783, 0.2728]; ROC-AUC: LR 0.9418 ± 0.0124, LightGBM 0.9505 ± 0.0122, Oracle 0.9577 ± 0.0114; Median train latency: LR 1777.36 ms vs LightGBM 155.00 ms), npm run test:all (23 pytest passed in 5.39s; test_generator.js PASS; test_causal.js PASS; verify_parity.js PASS: max diff 2.220e-16 < 1e-9).
- Suggested commit: feat(tests): implement multi-seed benchmark with bootstrap CIs, generative oracle ceiling, and markdown/json reporting

- What changed: In backend/lambda_function.py, enforced x-api-key authentication using hmac.compare_digest against env TRAIN_API_KEY for POST /train and /pipeline, failed closed with 500 when unset, kept GET /health, GET /model, and POST /predict public, and removed old optional API_KEY; in deploy/template.yaml, configured TrainApiKey parameter (NoEcho, MinLength 16, no default), removed '*' default from AllowedOrigin, added HttpApi DefaultRouteSettings (rate 5, burst 10) and stricter RouteSettings for POST /train, set Lambda ReservedConcurrentExecutions to 3, and scoped IAM policies for s3:PutObject on models/*, artifacts/*, pipeline_status.json and s3:GetObject on models/*; added TestTrainAuth pytest suite.
- Test results: npm run test:all (PASS: 29 pytest passed in 5.33s including 401 without key, 401 with invalid key, 200 with valid key, 500 fail closed when env unset, /pipeline protection, public endpoints; test_generator.js PASS: 50,000 records, 662 frauds / 1.324%; test_causal.js PASS: 50 cut points invariant; verify_parity.js PASS: max diff 2.220e-16 < 1e-9), compare_before_after.py PASS (105 trees, 0.9430 ROC-AUC, 0.2577 PR-AUC, 231.21 ms).
- Suggested commit: feat(security): enforce TRAIN_API_KEY auth on training routes, harden SAM template throttling, and scope IAM permissions

- What changed: Removed API_KEY from frontend/config.js and frontend/config.example.js, gitignored frontend/config.js, added password input "Training key" stored in sessionStorage in Stage 6, updated trainModelStage to send x-api-key on POST /train when key is entered, or fetch GET /model (setting modelSource to "Loaded pre-trained model from AWS (GET /model)") when no key is entered, updated prepareModelTrees and scoreRow to seamlessly handle unnested bundles directly from GET /model as well as nested bundles, preserved local fallback, and added missing config.js warning to deploy/deploy_s3.sh.
- Test results: npm run test:all (PASS: 29 pytest passed in 5.31s; test_generator.js PASS: 50,000 records, 662 frauds / 1.324%; test_causal.js PASS: 50 cut points invariant; verify_parity.js PASS: max diff 2.220e-16 < 1e-9), compare_before_after.py PASS (105 trees, 0.9430 ROC-AUC, 0.2577 PR-AUC, 234.17 ms).

- What changed: Created deploy/get_libgomp.sh for CloudShell/Docker to install libgomp (dnf/microdnf) and copy libgomp.so.1* to build/layer/lib/; updated deploy/build_layer.sh to delete python/bin/*.exe, preserve build/layer/lib/, and exit 1 if libgomp.so.1 is missing unless ALLOW_NO_LIBGOMP=1; updated deploy/template.yaml with CreateArtifactsBucket parameter (default 'true'), condition, and conditional bucket creation; removed manual aws s3 cp and publish-layer-version steps from README.md and deploy/LAYER.md in favor of SAM ContentUri layer packaging; created deploy/verify_live.sh curling /health and exiting 1 unless lightgbm_available is true; and added build/, data/, deploy/lambda_function.zip, and backend/tests/temp_parity_data.json to .gitignore.
- Test results: npm run test:all (PASS: 29 pytest passed in 5.42s; test_generator.js PASS: 50,000 records, 662 frauds / 1.324%; test_causal.js PASS: 50 cut points invariant; verify_parity.js PASS: max diff 2.220e-16 < 1e-9), compare_before_after.py PASS (105 trees, 0.9430 ROC-AUC, 0.2577 PR-AUC, 234.76 ms).
- Suggested commit: feat(deploy): automate layer packaging, add get_libgomp and verify_live scripts, and make artifacts bucket conditional

- What changed: Synchronized README.md and deploy/LAYER.md using only numbers from benchmark.md, benchmark.json, paysim.json, and FIXLOG.md: replaced benchmark table with multi-seed 5-seed aggregate results (PR-AUC 0.2238 vs 0.1608) and sample_payload test metrics (105 trees, 0.2577 PR-AUC), removed the 15.6x faster claim in favor of 11.5x (155.00 ms vs 1777.36 ms), replaced 2 MB limit with real limits (10 MB HTTP API, 6 MB synchronous Lambda), updated threshold derivation section and responses to match derive_thresholds (0.01 grid, 0.91 dec / 0.25 rev / 0.95 block, fallback, warnings), updated IAM scoping to match template.yaml, corrected balance-error formulas in feature table to match expected_balances code, converted file:/// links to relative, removed "The prompt investigated" from LAYER.md, added Limitations entries for synthetic formula / oracle ceiling (0.2220), post-transaction balance features / PaySim ablation (TODO), scale_pos_weight calibration, and noisy threshold sample variance, and updated viva Q&A for calibration (Q3), threshold derivation (Q9), and security (Q12).
- Test results: pytest (29 passed in 5.30s), test_generator.js (PASS: 50,000 records, 662 frauds / 1.324%), test_causal.js (PASS: 50 cut points invariant), verify_parity.js (PASS: max diff 2.220e-16 < 1e-9), compare_before_after.py (PASS: 105 trees, 0.9430 ROC-AUC, 0.2577 PR-AUC, 235.38 ms).
- Suggested commit: docs: update README and LAYER.md with multi-seed benchmark metrics, real limits, and limitations.
- What changed: Added scripts/prep_<name>.py for creditcard, elliptic, ethereum, bitcoinheist, and taiwan to standardize datasets to (header, label, optional order, numeric features) and generate stratified 300-row fixtures in tests/fixtures/real/, and added dataset/ to .gitignore.
- Test results: All 5 prep scripts executed successfully producing standardized datasets and 300-row samples (creditcard: 284,807 rows, 492 pos; elliptic: 46,564 rows, 4,545 pos; ethereum: 9,841 rows, 2,179 pos; bitcoinheist: 291,670 rows, 4,131 pos; taiwan: 30,000 rows, 6,636 pos; all 5 fixtures 300 rows with >=30 pos); npm run test:all (PASS: 29 pytest passed in 4.50s; test_generator.js PASS; test_causal.js PASS; verify_parity.js PASS).
- Suggested commit: feat(data): create dataset preparation scripts and stratified 300-row test fixtures for real fraud datasets
- What changed: Added Real dataset mode to frontend/index.html Stages 1-5 with mode toggle, incremental streaming CSV parser (file.stream().getReader() + TextDecoder({stream:true})), 300,000 row cap, Float64Array per-column storage, auto-detection for label/order dropdowns with 0/1 fallback, Stage 2 cleaning (missing % and constant column dropping), Stage 3 EDA (prevalence and Cohen's d separation), Stage 4 numeric passthrough (no leakage statement), Stage 5 chronological/seeded split (70% train / 30% test with 80% fit / 20% val), and disabled Stage 6 training button; added JS unit test suite in tests/test_real_mode.js; and wired into package.json test:js and test:all.
- Test results: node tests/test_real_mode.js (PASS: streaming parser chunk boundaries/quotes/blanks, 300k row cap, auto-detect across all 5 sample300 fixtures, chronological and mulberry32(42) splits); npm run test:all (PASS: 29 pytest passed in 4.68s, 4 JS suites passed including test_generator, test_causal, verify_parity, and test_real_mode); automated browser UI verification passed in Microsoft Edge with zero console errors.
- Suggested commit: feat(frontend): add Real dataset mode for Stages 1-5 with streaming parser, auto-detect, and chronological/seeded splitting

- What changed: Updated backend/lambda_function.py so POST /train works without test field, maps null to NaN, respects fit_rows in options, and extracts missing_type; updated frontend/index.html Stage 6 in real mode with buildRealTrainPayload (all fit positives kept, 1:1 mulberry32(42) undersampled negatives, natural val slice subsampled under 4.5 MB budget, test slice never sent, 6-digit rounding), POST /train with x-api-key, pure-JS test scoring with LightGBM NaN-as-0 predict handling, headline PR-AUC and ROC-AUC/F1/confusion matrix, POST /predict parity check, and real-mode notices on Stage 7 and dashboard; added sklearn metrics generator scripts/generate_metrics_fixture.py, real parity generator scripts/generate_real_parity_data.py, and new tests in backend/tests/test_model.py and tests/test_real_mode.js.
- Test results: pytest backend/tests/test_model.py backend/tests/test_feature_parity.py (PASS: 31 passed in 4.59s including POST /train without test field); node tests/test_real_mode.js (PASS: all 8 test suites passed including Test 6 JS-vs-Python PR-AUC/ROC-AUC against sklearn fixtures with ties within 2.78e-17 < 1e-9, Test 7 LightGBM parity on Part A sample with NaNs within 2.78e-17 < 1e-9, Test 8 payload builder 21.8 KB < 4.5 MB with 1:1 balance); npm run test:all (PASS: 31 pytest passed, 4 JS suites passed).
- Suggested commit: feat(real-mode): enable Stage 6 Lambda training without test field, pure-JS test scoring, parity with NaNs, and metrics validation

- What changed: Created scripts/run_real_eval.js and scripts/train_local_helper.py running the full real-mode pipeline headlessly using the exact browser JS functions and local Lambda handler, evaluated all 5 standardized datasets to generate results/real_eval.json, documented raw dataset download locations, preparation commands, standard CSV schema, pipeline design (row cap, split rules, class-balanced fit slice, thresholds from natural validation slice, validation subsampling under 4.5 MB budget, test isolation, and headline PR-AUC), and the measured results table in README.md, added any-schema support to Limitations, and cleaned up temporary evaluation files.
- Test results: scripts/run_real_eval.js evaluated all 5 datasets (creditcard: 284,807 rows, 0.17% prev, 16,415 train sent, PR-AUC 0.7522, ROC-AUC 0.9782, F1 80.60%; elliptic: 46,564 rows, 9.76% prev, fit slice alone 8.64 MB exceeds 4.5 MB budget, safely halted; ethereum: 9,841 rows, 22.14% prev, 3,747 train sent, PR-AUC 0.9953, ROC-AUC 0.9984, F1 97.24%; bitcoinheist: 291,670 rows, 1.42% prev, 44,765 train sent, PR-AUC 0.0115, ROC-AUC 0.5194, F1 2.67%; taiwan: 30,000 rows, 22.12% prev, 11,622 train sent, PR-AUC 0.5642, ROC-AUC 0.7811, F1 53.34%; results saved to results/real_eval.json); npm run test:all (PASS: 31 pytest passed in 4.57s; test_generator.js PASS; test_causal.js PASS; verify_parity.js PASS: max diff 2.220e-16 < 1e-9; test_real_mode.js PASS: all 8 test suites passed).
- Suggested commit: docs(real-eval): add headless real dataset benchmark runner, cross-domain evaluation results, and README documentation
- What changed: Enabled Stage 7 (stage7Real) and Dashboard (dashboardReal) in Real dataset mode as a held-out test replay reusing Stage 6 scores and model thresholds without cloud calls, showing action counts (Allow/Review/Block), triage breakdown with fraud capture shares, top-50 test rows table, KPI cards, score histogram by class, action bar chart, gain curve, and action filter; removed "not available for real datasets" errors for stages 7 and Dashboard; added Node unit tests in tests/test_real_mode.js; and created tests/verify_real_browser.js verifying Stages 1-7 and Dashboard via real UI clicks with zero console errors.
- Test results: npm run test:all passed (31/31 pytest passed in 4.67s; test_generator.js PASS; test_causal.js PASS; verify_parity.js PASS: max diff 2.220e-16; test_real_mode.js PASS: action counts sum to 120 test rows, cumulative gain curve ends at 100%/100%, top-50 table sorted descending by score, verified on creditcard_sample300 fixture; tests/verify_real_browser.js PASS: full UI button-click traversal Stages 1-7 and Dashboard with 0 console errors in Microsoft Edge).
- Suggested commit: feat(real-mode): enable Stage 7 and Dashboard held-out replay with gain curve, top-50 table, and headless browser click verification
- What changed: Added Live Replay Stream panel to real-dataset Dashboard (dashboardReal) with Start/Pause/Reset controls, speed selector (1, 5, 20 rows/s), incremental test row replay appending at top of DOM table (capped at 100 rows), live counters (replayed, Allow, Review, Block, frauds caught), progress indicator and held-out replay disclaimer, 3-second block toasts with short fade (max 3, collapsed into "+N more blocked", prefers-reduced-motion respected, timer cleanup on pause/reset/nav/run), added Test 10 counter invariant unit test in tests/test_real_mode.js, and updated tests/verify_real_browser.js to test ethereum_sample300.csv with incremental arrival, toast display/removal after 3.5s, pause, reset, and zero console errors.
- Test results: npm run test:all passed (31/31 pytest passed in 4.62s; test_generator.js PASS; test_causal.js PASS; verify_parity.js PASS: max diff 2.220e-16; tests/test_real_mode.js PASS including Test 10 asserting 90 replayed = 72 Allow + 0 Review + 18 Block with 9 frauds caught; tests/verify_real_browser.js PASS: Edge headless browser clicked Stages 1-7 and Dashboard with ethereum_sample300.csv, incremental rows 1 -> 3, Block toast on Row #211 score 0.9556 removed after 3.5s, Pause held row count at 3, Reset cleared table and counters to 0, 0 console errors).
- Suggested commit: feat(real-dashboard): add live replay stream panel with block toast notifications, speed controls, and headless browser tests
