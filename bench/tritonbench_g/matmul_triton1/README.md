# matmul_triton1

- Source file: `matmul_triton1.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

`FPCorrect.lean` now supplies two contracts for the complete accepted K loop
and final output store. They retain `BLOCK_K>0`, `K=BLOCK_K*numKBlocks`,
`BLOCK_N<=n_size`, the original program-ID addressing and all loop iterations.
There is no added separation premise: input and output may alias, since the
entire K loop finishes reading before the output store. Every output value is
specified from initial memory, and every other cell retains its initial value.

The new source uses actual fp32 input, weight and output storage channels,
matching the local Python test's floating profile. It retains
`allow_tf32=False`, a two-operand dot, and the separate accumulator addition.
A same-format fp32 cast on the dot result resolves the frontend's parameterized
dtype boundary; the backend proves this cast preserves bits. The generated
zero literal uses the `.real` callback tag before conversion to fp32, but its
payload is the fp32 positive-zero bit pattern; no Real evaluator is used.
`FPShape.source_projection` checks the entire original source after dtype
erasure, including the loop and dot options. The original mathematical source
and all its public statements remain unchanged.

The first public contract exposes the zero, dot and add primitives. The second
implements binary32 multiplication and addition with an explicit valid
reduction tree for each dot. Every input product and explicit zero seed remains
in the tree, and the outer accumulator addition rounds separately. This is a
parameterized software bit contract for arbitrary loop counts, inputs and
launch IDs. It does not identify that tree or its unfused operations with the
native compiler's dot instruction schedule. `FPWitness` checks three K blocks,
a nonzero program ID, an input/output alias, row gaps and a nonidentity fp32
rounding case; this small software regression does not stand in for the
parameterized theorem or a legal GPU launch test.

Native obligations remain: the actual IEEE/TF32 compiler configuration,
dot reduction/fusion schedule, fp32 rounding/subnormal/NaN policy, typed storage,
address widths, allocation bounds and launch legality. The Python host function
does not constrain every possible caller dtype; other input/output formats are
not covered by this fp32 profile. There was no GPU execution or new full-library
build for this addition.
