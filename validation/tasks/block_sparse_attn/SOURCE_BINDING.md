# Source and compiler binding

Unchanged original Python SHA256:
`c7e6c7bbfc3363b542cf52083f9dde7feb7ca38c3801b75166235d008d76d30b`.

Pinned repaired JIT function AST SHA256, excluding location attributes:
`3d61b6f42ebad945a3b40313608a63e37ef669ecffc9a55217eb6398d1e4f9d1`.
The reproducible source audit checks this exact digest. The JIT function is
unchanged from the historical full-channel repair. The new wrapper narrows the
selected specialization and makes its integer/storage contract executable.

| Source operation | Proof or consumed graph fragment |
| --- | --- |
| Three-dimensional grid, channel ownership | Source.work, lane_coverage, channel_coverage |
| Q/K/V pointer arithmetic and masks | Source query/key/value pointer equalities; CheckedCSR.data_contract; Physical query/key_value_load |
| Signed row endpoints, column slots, tile offsets | CheckedCSR.Guard → data_contract → Safety.all_data_nodes_exact |
| Every score channel block | SourceBinding.scoreDot and score_map; Spec.scoreLoop_full |
| CSR occurrence count and key order | SourceBinding.count/count_map and key/key_map |
| Scale and causal mask | SourceBinding.block; selected score FMA with zero/negative-infinity branch |
| Max/sum and online normalization | SourceBinding.update/update_map; normalizedStep explicit ordered tree/owner graph |
| FP32 P cast, live seeded PV | Same-format identity; fixed-eight-pair MMA chain with live seed |
| Empty/reversed row | Zero initial accumulator with zero iterations |
| All-masked visited tile | Native exceptional-word recurrence retains NaNs |
| Final channel stores | SourceBinding.StoreMap consumed by selectedExecute/selectedDispatch; physical readback and complete byte frame |

`SourceBinding.Binding` is an external selected-specialization correspondence,
separate from local numerical instruction laws. Its fields are local graph
maps and are consumed by the public execution. No field asserts a kernel
result, a target tensor, or execution equality. The source pin records which
Python function the external correspondence concerns; it does not prove the
correspondence. There is no claim of a full repaired-AST register invariant
or a verified compiler.

The selected profile is FP32, B_M=B_N=B_D=16, four warps, three stages,
Triton 3.7.1 targeting SM90. Saved profiles cover channel-block counts 1–4 and
both EVEN_M/EVEN_N settings together. They establish observed instruction
families and provide reviewable PTX/TTGIR; they do not universally establish
every operand route, owner or reduction tree for all admitted calls.

Evidence: `/tmp/veritile-block-sparse-check-final.log`,
`/tmp/veritile-block-sparse-lowering.log`,
`/tmp/veritile-block-sparse-lowering/report.json`, and the final complete
22-module build `/tmp/veritile-block-sparse-final-build.log`.
