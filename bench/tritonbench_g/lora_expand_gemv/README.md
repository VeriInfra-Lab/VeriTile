# lora_expand_gemv

- Source file: `lora_expand_gemv.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The new `FPCorrect.lean` contract executes the complete accepted active body:
`ADD_INPUTS=false`, `CAST_TYPE=false`, the masked input load, and every iteration
of `range(0, split_n_length, BLOCK_N)`. It covers arbitrary output length and the
last partial tile, with the original `BLOCK_N>0`, `K<=BLOCK_K`, distinct output
and weight regions, and positive output stride. Input/output aliasing is allowed:
the entire input vector is cached before the first output write. LoRA-index
metadata is also captured from the initial state. Every active output cell and
all other memory are specified independently of execution.

Four input profiles cover half/half, half/bfloat16, bfloat16/half, and
bfloat16/bfloat16. Three explicit floating output formats cover half, bfloat16,
and fp32. The source tracks their storage channels; half and bfloat16 payloads
occupy the low sixteen bits of the carrier. The implicit store conversion is
materialized as an explicit cast. The original mathematical theorem did not
select a storage format; this result does not claim all possible Python output
dtypes or the host dispatch/signed-sentinel path.

The reviewed local Triton 3.7.1 `standard.sum` and `core.reduce` retain the floating input dtype. The earlier 3.8 version label was incorrect.
Half/bfloat16 mixed multiplication converts the bfloat16 operand to half; sum
then retains that product dtype. This corrects the earlier planning review's
unverified suggestion of automatic fp32 sum promotion. `FPModel` implements true
binary16 and bfloat16 products, additions, and direct cross-format casts. The
`.real` callback channel is used only for the generated zero-padding literal;
it carries fp32 bits and does not invoke a mathematical Real evaluator.

The public result quantifies a valid `ReductionPlan`: every `BLOCK_K` product
and every explicit zero seed remains in the tree. No padding cancellation,
reassociation, finite-input condition, or single-tile restriction is used.
The selected tree and scalar configuration require a separate compiler/device
binding, including NaN/zero behavior, rounding and FTZ, low-precision arithmetic,
output casts, and possible compiler fusion. `FPWitness` supplies a supporting
three-iteration run with input/output aliasing, K padding and an output tail;
it does not replace the parameterized contract.


`BranchCorrect.lean` adds the complete per-program source: the signed `-1`
sentinel, both `EVEN_K`, `ADD_INPUTS` and `CAST_TYPE` values, and every output-loop
iteration. Six input/weight pairs (fp16, bf16 or fp32 input; fp16 or bf16
weights) combine with fp16, bf16 or fp32 output. Only `BLOCK_N > 0` is assumed.
The explicit software engine binds all used arithmetic, including the fp32
real-channel reduction. There is no numerical callback premise.

The independent reference reads each output block from that block's entry
memory, so later blocks observe earlier stores and weight/output aliases.
The full memory result and frame require no alias or output-injectivity
hypothesis. This is ordered interpreter behavior; it does not establish
race safety for colliding native stores. A full-source projection proves the
entire original body after dtype erasure and constant-branch normalization,
retaining the dynamic sentinel and all loops. Concrete checks exercise fp32
casting, add-inputs, aliases, padding, the final tail and the signed sentinel.

This is still a selected software execution profile. The wrapper's squeeze,
shape/stride/grid dispatch, other output/index dtypes, admissible native tile
shapes, signed fixed-width address and byte-memory correspondence, and native
concurrency remain structural obligations. They cannot be deferred as numerical
assumptions. Floating configuration, reduction-tree/fusion and conversion
bindings to the compiler/device require separate numerical follow-up.
