#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

targets=()
for example in VectorAdd RowWiseSum LogExp; do
  python3 -B -m scripts.bench.generate_jit_correspondence "bench/examples/${example}"
  targets+=("bench.examples.${example}.RealCorrect" "bench.examples.${example}.FPEquiv")
done

lake build "${targets[@]}"
