# AWS Lambda Layer Architecture & Packaging Guide

## Overview

This guide provides instructions and technical analysis for packaging and deploying the **LightGBM + NumPy** runtime layer for the **Serverless Financial Fraud Detection Platform**.

* **Target Runtime:** Python 3.12 (`python3.12`)
* **Architecture:** `x86_64` (Amazon Linux 2023 base)
* **Layer Contents:** `lightgbm`, `numpy`, `scipy`, `narwhals` (+ optional `libgomp.so.1` in `lib/`)
* **Build Script:** [`build_layer.sh`](build_layer.sh)

---

## 1. Dependency Analysis: Why SciPy and Narwhals are Required

We investigated whether `lightgbm` can run in a minimal layer containing only `lightgbm` + `numpy` without `scipy`.

### Empirical Findings:
Inspection of `lightgbm/__init__.py` and `lightgbm/basic.py` reveals:
```python
# lightgbm/basic.py line 28-31
import narwhals as nw
import narwhals.dependencies as nwd
import narwhals.typing as nwt
import numpy as np
import scipy.sparse
```
When attempting to import LightGBM without `scipy`:
```text
ModuleNotFoundError: No module named 'scipy.sparse'; 'scipy' is not a package
```
**Conclusion:** LightGBM 4.x unconditionally imports `scipy.sparse` and `narwhals` at module initialization. Therefore, **`scipy` and `narwhals` must be included in the layer**.

---

## 2. Dynamic Linker & Shared Library Audit (libgomp.so.1)

### ELF Header Inspection of `lib_lightgbm.so`
Using automated ELF parser on `lightgbm-4.7.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`:
```text
File: lightgbm/lib/lib_lightgbm.so
DT_NEEDED Shared Library Dependencies:
  - libdl.so.2
  - libstdc++.so.6
  - libm.so.6
  - libgomp.so.1        <-- OpenMP Runtime (CRITICAL)
  - libgcc_s.so.1
  - libpthread.so.0
  - libc.so.6
  - ld-linux-x86-64.so.2
```

### The OpenMP Risk in Amazon Linux 2023 (AL2023)
AWS Lambda's Python 3.12 managed runtime is based on a stripped-down Amazon Linux 2023 minimal container image. Unlike older Amazon Linux 2 runtimes, `libgomp.so.1` is **not guaranteed to be installed in the default Lambda container root filesystem**.

If `libgomp.so.1` is missing, invoking `import lightgbm` will fail with:
```text
OSError: libgomp.so.1: cannot open shared object file: No such file or directory
```

### Remediation: Supplying `libgomp.so.1` via `/opt/lib`
In AWS Lambda, any directory named `lib/` in a layer is mounted at `/opt/lib`, which AWS Lambda automatically includes in `LD_LIBRARY_PATH`.

To bundle `libgomp.so.1` into the layer:
1. Extract `libgomp.so.1` from an Amazon Linux 2023 container environment:
   ```bash
   mkdir -p build/layer/lib
   docker run --rm -v "$PWD/build/layer/lib:/out" public.ecr.aws/lambda/python:3.12 \
     bash -c "dnf install -y libgomp && cp -P /usr/lib64/libgomp.so.1* /out/"
   ```
2. Alternatively, copy `libgomp.so.1` from an auditwheel-repaired manylinux wheel.
3. If Docker is unavailable on the local build machine:
   > [!NOTE]
   > **UNVERIFIED on current Windows machine without Docker:**
   > Run the command above on any Linux machine or AWS CloudShell, place `libgomp.so.1` into `build/layer/lib/`, and execute `./deploy/build_layer.sh`.

### Diagnostics in `/health`
To verify layer integrity live on AWS without reading CloudWatch logs, the Lambda `/health` route reports:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "service": "aws-financial-fraud-detection",
  "model_available": true,
  "lightgbm_available": true,
  "lightgbm_version": "4.7.0",
  "lightgbm_error": null
}
```
If the layer is missing or `libgomp.so.1` is not found, `lightgbm_available` reports `false` with the exact `OSError` or `ModuleNotFoundError` text.

---

## 3. Platform Tag: manylinux2014 vs manylinux_2_28

When using `pip install --platform`:
* `--platform manylinux2014_x86_64` targets glibc 2.17. LightGBM 4.x discontinued manylinux2014 for x86_64; pip will downgrade to legacy LightGBM 3.3.5.
* `--platform manylinux_2_28_x86_64` targets glibc 2.28. Amazon Linux 2023 uses glibc 2.34, which is backwards-compatible with `manylinux_2_28`. This installs current LightGBM 4.7.0.

`build_layer.sh` defaults to `manylinux_2_28_x86_64`.

---

## 4. Size Limits & S3 Upload Procedure

AWS Lambda enforces strict size limits on layers:
1. **Direct Console Zip Upload Limit:** **50 MB** (zipped).
2. **Total Layer Unzipped Quota:** **250 MB** (unzipped across all attached layers).

### Trimming Strategy (implemented in `build_layer.sh`):
* Deletes `__pycache__` directories.
* Deletes `tests/` and test suites from `numpy`, `scipy`, and `lightgbm`.
* Deletes `*.pyi` type stub files.
* Strips debug symbols from native `.so` files using `strip --strip-unneeded`.

### Size Outcomes:
* **Unzipped size after trimming:** $\approx 150 - 180 \text{ MB}$ (well within the $250 \text{ MB}$ quota).
* **Zipped size:** $\approx 45 - 55 \text{ MB}$.

### Automatic Layer Upload via AWS SAM:
While direct upload through the AWS Console web UI has a 50 MB limit, AWS SAM automatically packages and uploads the layer zip directly to the deployment S3 bucket via `ContentUri: ../build/layer/lightgbm-layer.zip`. Manual `aws s3 cp` and `publish-layer-version` commands are unnecessary.

---

## 5. SAM Template Integration

In [`template.yaml`](template.yaml), configure:
```yaml
Resources:
  LightGbmLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: LightGBM-Python312-Layer
      Description: LightGBM and NumPy binary layer for Python 3.12
      ContentUri: ../build/layer/lightgbm-layer.zip
      CompatibleRuntimes:
        - python3.12
      CompatibleArchitectures:
        - x86_64
      RetentionPolicy: Retain

  FraudPipelineLambda:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: FraudPipelineRunner
      CodeUri: ../backend
      Handler: lambda_function.lambda_handler
      Runtime: python3.12
      Architectures:
        - x86_64
      MemorySize: 2048
      Timeout: 60
      Layers:
        - !Ref LightGbmLayer
```
