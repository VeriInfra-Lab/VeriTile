# Tracked KV cache proof reconstruction

The current selected copy/INT8/INT4 kernel proof chain is reconstructed as
tracked Lean source. Both public entries build, pass their standard-base axiom
checks, and the actual original/candidate FP theorem prints no experimental
assumptions. This is a new source delivery, not a recovery of old cache bytes.

`review.json` records the direct semantic review, source commit and validation.
`source-closure.json` pins the complete local import closure. The new clean
checkout has no project `.lake/build` and no `.veritile-cache`; pinned external
Lean/Mathlib dependencies reuse their installed artifacts. The clean build passed all 8,712 jobs; the compressed log and its hash are
recorded in `review.json`.

The reconstruction covers actual source access safety, successful execution,
all cache and scale/zero outputs and exact exterior memory frames. The Real
formula uses full logical-head extrema, preserves the constant-row scale case
and paired INT4 packing, and is independent of the source evaluator. The FP
pair preserves all primitive calls and changes only the integer key-address
product. Named source/destination/key/value inputs and all four typed outputs
are bijective views of the existing memory contract; `judgment_iff` checks the
complete judgment connection.

The 183-task source inventory found the selected missing source concentrated
in this task. It is an availability inventory, not a new proof audit of all
183 tasks. Fresh Llama source-safety and public-entry builds passed; their logs
are included. `matmul_leakyrelu` retains its separate numerical deferral.
The subsequent all-selected source replay and combined-import results are
recorded separately in [selected-source-replay](../selected-source-replay/README.md).

`superseded-selection.json` preserves the previous selected paths, hashes,
review rows and delivery description. Seven optional historical modules
(ABI/raw/Tensor/backing-memory results) remain unavailable; they are listed in
`review.json`. The new main proof does not import them or renew their old
acceptance. The earlier failure checkpoint under `../source-closure/` remains
an accurate record of the earlier commit.

The official comparator is unavailable in this environment; its failed launch
is retained in `comparator.log`. No GPU or new numerical experiment ran.
