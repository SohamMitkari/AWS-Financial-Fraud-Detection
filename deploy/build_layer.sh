#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AWS Lambda Layer Packaging Script: LightGBM + NumPy (+ Scipy + Narwhals)
# Target Runtime: Python 3.12, Amazon Linux 2023 (x86_64)
# No Docker required: uses pip wheel platform targeting
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${PROJECT_ROOT}/build/layer"
TARGET_DIR="${BUILD_DIR}/python"
LIB_DIR="${BUILD_DIR}/lib"
ZIP_OUTPUT="${BUILD_DIR}/lightgbm-layer.zip"

echo "=== Building AWS Lambda Layer for Python 3.12 (x86_64) ==="
echo "Cleaning target directory ${TARGET_DIR} and previous zip..."
rm -rf "${TARGET_DIR}" "${ZIP_OUTPUT}"
mkdir -p "${TARGET_DIR}"
mkdir -p "${LIB_DIR}"

# 1. Platform Wheel Selection:
# Amazon Linux 2023 provides glibc 2.34 (manylinux_2_28_x86_64).
# manylinux2014 corresponds to glibc 2.17.
PLATFORM="${PLATFORM:-manylinux_2_28_x86_64}"

echo "Installing wheel dependencies for platform '${PLATFORM}'..."
pip install \
  --platform "${PLATFORM}" \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all: \
  --target "${TARGET_DIR}" \
  lightgbm numpy scipy narwhals

# 2. OpenMP (libgomp) runtime check:
# If libgomp.so.1 was placed directly in build/layer/, copy to lib/
if [ -f "${BUILD_DIR}/libgomp.so.1" ]; then
  echo "Found libgomp.so.1 in build directory. Copying to layer lib/ (/opt/lib on Lambda)..."
  cp "${BUILD_DIR}/libgomp.so.1"* "${LIB_DIR}/"
fi

# Exit 1 if libgomp.so.1 is missing unless ALLOW_NO_LIBGOMP=1
if [ ! -f "${LIB_DIR}/libgomp.so.1" ] && [ ! -L "${LIB_DIR}/libgomp.so.1" ]; then
  if [ "${ALLOW_NO_LIBGOMP:-0}" = "1" ]; then
    echo "WARNING: ${LIB_DIR}/libgomp.so.1 is missing, but ALLOW_NO_LIBGOMP=1 is set. Proceeding..."
  else
    echo "ERROR: Missing ${LIB_DIR}/libgomp.so.1. LightGBM requires libgomp on Lambda AL2023." >&2
    echo "Run deploy/get_libgomp.sh first or set ALLOW_NO_LIBGOMP=1 to bypass." >&2
    exit 1
  fi
fi

# 3. Trimming bloated non-runtime files:
echo "Trimming non-runtime files (__pycache__, tests, *.pyi, documentation, executables)..."
find "${TARGET_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${TARGET_DIR}" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "${TARGET_DIR}" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find "${TARGET_DIR}" -name "*.pyi" -delete 2>/dev/null || true
find "${TARGET_DIR}" -name "*.pyc" -delete 2>/dev/null || true

# Delete python/bin/*.exe
rm -f "${TARGET_DIR}/bin/"*.exe 2>/dev/null || true
if [ -d "${TARGET_DIR}/bin" ]; then
  find "${TARGET_DIR}/bin" -name "*.exe" -delete 2>/dev/null || true
fi

# Strip native shared objects if strip utility is available
if command -v strip >/dev/null 2>&1; then
  echo "Stripping unneeded debug symbols from native shared objects..."
  find "${TARGET_DIR}" -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true
  if [ -d "${LIB_DIR}" ]; then
    find "${LIB_DIR}" -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true
  fi
fi

# 4. Packaging the Lambda Layer:
# The top-level directory inside the zip MUST be `python/` (and optionally `lib/`)
echo "Archiving layer to ${ZIP_OUTPUT}..."
(
  cd "${BUILD_DIR}"
  if [ -d "lib" ] && [ "$(ls -A lib 2>/dev/null)" ]; then
    zip -q -9 -r "${ZIP_OUTPUT}" python lib
  else
    zip -q -9 -r "${ZIP_OUTPUT}" python
  fi
)

# 5. Report artifact sizes:
ZIPPED_BYTES=$(stat -c%s "${ZIP_OUTPUT}" 2>/dev/null || stat -f%z "${ZIP_OUTPUT}")
UNZIPPED_BYTES=$(du -sb "${TARGET_DIR}" 2>/dev/null | awk '{print $1}' || du -sk "${TARGET_DIR}" | awk '{print $1 * 1024}')

ZIPPED_MB=$(awk "BEGIN {printf \"%.2f\", ${ZIPPED_BYTES} / (1024*1024)}")
UNZIPPED_MB=$(awk "BEGIN {printf \"%.2f\", ${UNZIPPED_BYTES} / (1024*1024)}")

echo "=========================================================="
echo "Layer build complete!"
echo "Artifact path:    ${ZIP_OUTPUT}"
echo "Zipped size:      ${ZIPPED_MB} MB (${ZIPPED_BYTES} bytes)"
echo "Unzipped size:    ${UNZIPPED_MB} MB (${UNZIPPED_BYTES} bytes)"
echo "Direct upload:    $([ $(awk "BEGIN {print (${ZIPPED_BYTES} <= 52428800)}") -eq 1 ] && echo "YES (<= 50 MB console limit)" || echo "NO (> 50 MB, upload via S3)")"
echo "Unzipped limit:   $([ $(awk "BEGIN {print (${UNZIPPED_BYTES} <= 262144000)}") -eq 1 ] && echo "PASS (<= 250 MB quota)" || echo "EXCEEDS QUOTA (> 250 MB)")"
echo "=========================================================="
