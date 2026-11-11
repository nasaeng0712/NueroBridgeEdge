#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "과학·모델 경로에는 Python 3.11 이상이 필요합니다.")'
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-model.txt
