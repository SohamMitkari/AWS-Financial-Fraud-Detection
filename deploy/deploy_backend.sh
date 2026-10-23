#!/usr/bin/env bash
set -euo pipefail

# Package and update AWS Lambda function code
# Usage:
#   LAMBDA_FUNCTION_NAME=FraudPipelineRunner [AWS_REGION=ap-southeast-2] ./deploy/deploy_backend.sh

FUNCTION_NAME="${1:-${LAMBDA_FUNCTION_NAME:-FraudPipelineRunner}}"
REGION="${AWS_REGION:-ap-southeast-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BACKEND_DIR="${PROJECT_ROOT}/backend"
ZIP_PATH="${SCRIPT_DIR}/lambda_function.zip"

echo "Packaging Lambda deployment package from ${BACKEND_DIR}..."
rm -f "${ZIP_PATH}"

# Archive pure Python lambda_function.py
(cd "${BACKEND_DIR}" && zip -q -9 "${ZIP_PATH}" lambda_function.py)

echo "Updating function code for '${FUNCTION_NAME}' in region '${REGION}'..."
aws lambda update-function-code \
  --function-name "${FUNCTION_NAME}" \
  --zip-file "fileb://${ZIP_PATH}" \
  --region "${REGION}"

rm -f "${ZIP_PATH}"
echo "Backend Lambda deployment complete."
