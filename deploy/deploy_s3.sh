#!/usr/bin/env bash
set -euo pipefail

# Deploy frontend assets to Amazon S3 Static Website Hosting
# Usage:
#   ./deploy/deploy_s3.sh [BUCKET_NAME]
#   or: FRONTEND_BUCKET_NAME=my-bucket ./deploy/deploy_s3.sh

BUCKET_NAME="${1:-${FRONTEND_BUCKET_NAME:-}}"

if [[ -z "${BUCKET_NAME}" ]]; then
  echo "Error: S3 bucket name must be provided as the first argument or via FRONTEND_BUCKET_NAME environment variable." >&2
  echo "Example: ./deploy/deploy_s3.sh my-fraud-frontend-bucket" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

if [[ ! -f "${PROJECT_ROOT}/frontend/config.js" ]]; then
  echo "Warning: frontend/config.js is missing! Static frontend will use placeholder endpoints unless configured." >&2
fi

echo "Syncing frontend static assets from ${PROJECT_ROOT}/frontend to s3://${BUCKET_NAME}..."
aws s3 sync "${PROJECT_ROOT}/frontend" "s3://${BUCKET_NAME}/" --delete

echo "Frontend deployment complete. Access via your S3 Static Website endpoint."