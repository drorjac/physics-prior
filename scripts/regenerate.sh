#!/usr/bin/env bash
# Regenerate everything downstream of the source: results, figures, docs,
# notebooks, and then verify with the test suite.
#
#   ./scripts/regenerate.sh [--quick]
#
# Uses whatever `physprior` is on PATH, so activate the environment first.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

QUICK="${1:-}"

echo "=== physprior run all ${QUICK} ==="
physprior run all ${QUICK}

echo "=== physprior figures ==="
physprior figures

echo "=== physprior neglected ${QUICK} ==="
physprior neglected ${QUICK}

echo "=== physprior report ==="
physprior report

echo "=== physprior notebooks --execute ==="
physprior notebooks --execute

echo "=== tests ==="
python -m pytest -m "not slow and not network and not sr" -q

echo "=== REGENERATE DONE ==="
