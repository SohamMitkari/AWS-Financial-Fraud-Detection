#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Live Health & LightGBM Verification Script
# Usage: ./deploy/verify_live.sh <API_URL>
# Example: ./deploy/verify_live.sh https://xyz.execute-api.ap-southeast-2.amazonaws.com
# Calls /health and exits 1 unless lightgbm_available is true.
# ==============================================================================

API_URL="${1:-${API_URL:-}}"

if [ -z "${API_URL}" ]; then
  echo "Usage: $0 <API_URL>" >&2
  echo "Example: $0 https://xyz123.execute-api.ap-southeast-2.amazonaws.com" >&2
  exit 1
fi

# Strip trailing slash if present
API_URL="${API_URL%/}"
HEALTH_ENDPOINT="${API_URL}/health"

echo "Verifying live endpoint: ${HEALTH_ENDPOINT}"

# Fetch health response
RESPONSE=$(curl -fsSL --max-time 15 "${HEALTH_ENDPOINT}" 2>&1) || {
  echo "ERROR: Failed to connect to ${HEALTH_ENDPOINT}" >&2
  echo "${RESPONSE}" >&2
  exit 1
}

echo "Response payload: ${RESPONSE}"

# Check lightgbm_available == true
IS_LIGHTGBM_AVAILABLE="false"

if command -v python3 >/dev/null 2>&1; then
  IS_LIGHTGBM_AVAILABLE=$(python3 -c "import sys, json; data = json.loads(sys.argv[1]); print(str(data.get('lightgbm_available', False)).lower())" "${RESPONSE}" 2>/dev/null || echo "false")
elif command -v python >/dev/null 2>&1; then
  IS_LIGHTGBM_AVAILABLE=$(python -c "import sys, json; data = json.loads(sys.argv[1]); print(str(data.get('lightgbm_available', False)).lower())" "${RESPONSE}" 2>/dev/null || echo "false")
elif command -v node >/dev/null 2>&1; then
  IS_LIGHTGBM_AVAILABLE=$(node -e "try { const d = JSON.parse(process.argv[1]); console.log(Boolean(d.lightgbm_available)); } catch(e) { console.log('false'); }" "${RESPONSE}" 2>/dev/null || echo "false")
else
  if echo "${RESPONSE}" | grep -q '"lightgbm_available"[[:space:]]*:[[:space:]]*true'; then
    IS_LIGHTGBM_AVAILABLE="true"
  else
    IS_LIGHTGBM_AVAILABLE="false"
  fi
fi

if [ "${IS_LIGHTGBM_AVAILABLE}" != "true" ]; then
  echo "ERROR: lightgbm_available is not true in health check response!" >&2
  exit 1
fi

echo "SUCCESS: Live API is healthy and lightgbm_available is true."
exit 0
