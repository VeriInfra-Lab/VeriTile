# matmul_triton2

- Source file: `matmul_triton2.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The new `FPCorrect.lean` contracts cover the complete previously accepted
`K=BLOCK_K*numKBlocks` stage: the grouped program mapping, all six strides,
every iteration of the dynamic `cdiv(K,BLOCK_K)` loop, the original K-load
masks, and the final M/N output mask. The generic-stride contract retains the
original output-address injectivity condition; the row-major contract retains
`stride_cn=1` and `BLOCK_N<=stride_cm`. Both retain `BLOCK_K>0`. Input, weight
and output regions may alias: every K-loop read occurs before the output store.
Every active result comes from an independent initial-memory reference, and
all unwritten cells, including masked output lanes, retain their initial value.
The ragged-K extension of the Python kernel was outside the old accepted scope
and is not claimed here.

The explicit storage profile is fp32, matching the local Python tests. The
source keeps dot options unspecified (`{}`); it does not add
`allow_tf32=False`. A same-format fp32 cast materializes the frontend's generic
dtype boundary and preserves bits. The zero initializer's generated `.real`
callback tag carries fp32 positive-zero bits rather than a mathematical Real
value. The full source projection is checked after dtype erasure, and the old
mathematical Lean source and its public statements are unchanged.

There are three explicit software input modes: full binary32, TF32-grid
round-to-nearest ties-to-even, and TF32-grid round-to-nearest ties-away. The
latter two quantize away thirteen fraction bits before binary32 multiplication;
their encoded magnitude lies on that grid and within half a grid step.
Nonfinite handling, signed zero, midpoint differences and overflow are checked.
No mode is inferred from unspecified native dot options. The generic dot
contract also permits a separately bound input-projection implementation.
Every mode retains all products and explicit zero seeds in a supplied valid
reduction tree, followed by a separate rounded fp32 accumulator addition.

Native obligations include the target's actual default input precision, the
Triton `fp32_default` knob, TF32 conversion rounding/subnormal/NaN behavior,
the dot instruction's multiplication/accumulation precision and reduction
schedule, possible contraction of the outer addition, typed storage, address
widths, allocation bounds and launch legality. A full-fp32 software instance is
not asserted to match the usual TF32-capable target's default. Other caller
dtypes accepted by the untyped Python host interface are not covered by this
fp32 profile. No GPU or offline native compilation was run for this addition.

`FPWitness` executes three K blocks at a nonzero program ID in the last partial
M group, with nontrivial strides, input/output aliasing and both masked output
dimensions. It distinguishes the full-fp32 and TF32 software results and checks
that masked aliased cells retain their original bits. This small regression
supports the parameterized contracts; it is not a native launch test.

One source-level memory boundary is explicit: A/B loads mask only K, so partial
M/N output tiles still load complete input tiles. The new reference preserves
those reads. For logical 16×16 inputs, the listed autotuning tiles start at
32 rows/columns and therefore contain logical-shape-exterior input addresses.
The total-memory value contract does not prove those addresses allocated or
safe on a native launch; original allocation-window conditions still need to
be discharged. No input-padding premise or new input mask was added to the
public floating statements.
