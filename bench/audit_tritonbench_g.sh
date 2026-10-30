#!/usr/bin/env bash
# Validate the final selected tree. Source review, Lean compilation and proof
# replay remain independent gates; none certifies native GPU execution.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1-}"
if [ "$#" -gt 1 ] || { [ -n "$mode" ] && [ "$mode" != '--global-only' ] && [ "$mode" != '--source-only' ]; }; then
  printf 'Usage: %s [--global-only|--source-only]\n' "$0" >&2
  exit 2
fi
if [ -n "$mode" ] && { [[ -v AUDIT_TRUST_SHARD_COUNT ]] || [[ -v AUDIT_TRUST_SHARD_INDEX ]]; }; then
  printf 'Source/global audit mode cannot be combined with corpus shard configuration\n' >&2
  exit 2
fi
failures=0
python3 -B bench/final_inventory.py --check || failures=$((failures + 1))
python3 -B -m bench.check_body_corpus --check || failures=$((failures + 1))
python3 -B bench/check_imported_kernel_review.py || failures=$((failures + 1))
if [ "$mode" = '--source-only' ]; then
  [ "$failures" -eq 0 ] || exit 1
  printf 'Final source checks passed; no proof replay was requested\n'
  exit 0
fi
lake build VeriTile VeriTileMath VeriTileFull || failures=$((failures + 1))
python3 scripts/check_comparator.py --library || failures=$((failures + 1))
if [ "$mode" = '--global-only' ]; then
  [ "$failures" -eq 0 ] || exit 1
  printf 'TritonBench-G global gates passed; corpus audit is separate\n'
  exit 0
fi
lake build TritonBenchGFinal FinalRegressions TritonBenchSpecExamples BugDetection PretrainKernels || failures=$((failures + 1))
bench/audit_trust.sh --final-inventory || failures=$((failures + 1))
[ "$failures" -eq 0 ] || exit 1
printf 'TritonBench-G audit gates passed\n'
