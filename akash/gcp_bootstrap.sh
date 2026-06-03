#!/bin/bash
# MIRAGE GCP bootstrap — run on audit VM (no venv, system pip + --break-system-packages)
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HOME}/hf_cache"
export STATE_DIR="${HOME}/mirage-state"
export MIRAGE_ROOT="${HOME}/Audit_Benchmark"
export MIRAGE_CODE="${MIRAGE_ROOT}/Code/mirage"

mkdir -p "${HF_HOME}" "${STATE_DIR}"

if [[ ! -d "${MIRAGE_ROOT}/.git" ]]; then
  rm -rf "${MIRAGE_ROOT}"
  TOKEN="$(tr -d '\r\n' < "${HOME}/.github_creds")"
  git clone "https://${TOKEN}@github.com/DevDaring/Audit_Benchmark.git" "${MIRAGE_ROOT}"
  rm -f "${HOME}/.github_creds"
else
  cd "${MIRAGE_ROOT}" && git pull --ff-only
fi

echo "[bootstrap] git repo OK at ${MIRAGE_ROOT}"

cp "${HOME}/.env-mirage" "${MIRAGE_CODE}/.env"
chmod 600 "${MIRAGE_CODE}/.env"
sed -i 's/\r$//' "${MIRAGE_CODE}/.env"
sed -i '/^PHONE_NO/d;/^TextBelt/d' "${MIRAGE_CODE}/.env"
cd "${MIRAGE_CODE}"
set -a && source .env && set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN}"

python3 -c "from config import ensure_dirs; ensure_dirs(); print('ensure_dirs OK')"

echo "[bootstrap] clone and .env OK"
