#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Helper Script: Extract libgomp for AWS Lambda AL2023 Layer
# Works in AWS CloudShell or Docker container (public.ecr.aws/lambda/python:3.12)
# Installs libgomp (via dnf or microdnf) and copies libgomp.so.1* into build/layer/lib/
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
LIB_DEST="${PROJECT_ROOT}/build/layer/lib"
mkdir -p "${LIB_DEST}"

echo "=== Extracting libgomp for AWS Lambda (AL2023 / Python 3.12) ==="

if command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
elif command -v microdnf >/dev/null 2>&1; then
    PKG_MGR="microdnf"
else
    # If dnf/microdnf are not available on this host, check if Docker is available
    if command -v docker >/dev/null 2>&1; then
        echo "Neither dnf nor microdnf found on host. Running via Docker (public.ecr.aws/lambda/python:3.12)..."
        docker run --rm -v "${PROJECT_ROOT}:/workspace" -w /workspace public.ecr.aws/lambda/python:3.12 \
            bash -c "if command -v dnf >/dev/null 2>&1; then dnf install -y libgomp; elif command -v microdnf >/dev/null 2>&1; then microdnf install -y libgomp; fi && mkdir -p build/layer/lib && cp -P /usr/lib64/libgomp.so.1* build/layer/lib/ 2>/dev/null || cp -P /usr/lib/libgomp.so.1* build/layer/lib/"
        echo "Done. Extracted libgomp files to ${LIB_DEST}:"
        ls -la "${LIB_DEST}"
        exit 0
    else
        echo "Error: Neither dnf nor microdnf found, and docker is not available." >&2
        echo "Run this script inside AWS CloudShell or in Docker (public.ecr.aws/lambda/python:3.12)." >&2
        exit 1
    fi
fi

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

echo "Installing libgomp using ${PKG_MGR}..."
$SUDO "$PKG_MGR" install -y libgomp

echo "Copying libgomp.so.1* to ${LIB_DEST}/..."
COPIED=0
for f in /usr/lib64/libgomp.so.1* /usr/lib/libgomp.so.1*; do
    if [ -e "$f" ] || [ -L "$f" ]; then
        cp -P "$f" "${LIB_DEST}/"
        COPIED=1
    fi
done

if [ "$COPIED" -eq 0 ]; then
    find /usr -name "libgomp.so.1*" -exec cp -P {} "${LIB_DEST}/" \;
fi

if [ ! -e "${LIB_DEST}/libgomp.so.1" ]; then
    echo "Error: libgomp.so.1 was not found in ${LIB_DEST} after copy." >&2
    exit 1
fi

echo "Successfully placed libgomp into ${LIB_DEST}:"
ls -la "${LIB_DEST}"
