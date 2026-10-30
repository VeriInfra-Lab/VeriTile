Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

# sin_computation: checked FP32 kernel and index variant

The independent result is `out[i] = Real.sin(x[i])` for `i < x.numel()`;
remaining output cells are unchanged. The production profile is FP32 input,
FP32 `tl.sin`, and FP32 output. `Kernels.lean` contains the complete readable
original and candidate DSL. The retained candidate changes only
`pid * BLOCK_SIZE` to `BLOCK_SIZE * pid`. The public Real and FP statements
use these canonical sources.

Required acceptance covers the kernel. Python supplies its input assertions;
whole-grid scheduling, the wrapper's `None` return, and caller object identity
are outside that public proof scope. Their earlier proofs are retained as
optional results in `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`, and the
existing host helpers. The original benchmark Python remains unchanged.

`Misc/TensorBridge.lean` provides the optional Python-check correspondence. The main mathematical and FP declarations remain in `Correct.lean` and `FPEquiv.lean`; neither imports the bridge.

## Contract and memory meaning

Both arguments must be ordinary strided contiguous FP32 CUDA tensors on the same
device, without unresolved negative view bits. The output must have at least
as many elements as the input, whose count is below `2**31`. Shapes may differ.
Nonempty active spans need four-byte-aligned data pointers and must be exactly
identical or disjoint. Empty inputs impose neither pointer nor overlap checks.
No input-value finiteness guard is used; NaN/Inf, signed zeros, and subnormals
remain admitted.

`Contract.lean` names these checks and their raw active-byte-span meaning.
`TensorBridge.python_checks_iff_preconditions` proves exact dynamic admission
correspondence. The unchanged wrapper mutates only the first `n` output elements,
uses BLOCK4 and `ceil(n/4)` programs, enters the input device, and returns `None`.
It performs no allocation, input snapshot, copy, normalization, or layout change.

`Correct.sinIO X O n` declares the common active load/store mask and linear
windows. The Real theorem is generic in the region names: `X = O` represents
exact in-place aliasing, and distinct names have the ordinary disjoint flat
allocation meaning. The kernel loads the tile before storing it. Every
unwritten cell, including the output suffix, is framed. The theorem starts from
active-lane memory bounds; inactive tails do not require padding allocations.

## Original failures

The existing checks retain CPU witnesses for raw-pointer disagreement with
stride-two and unresolved-negative logical views. Shifted overlapping input and
output spans can race between programs, so the revised contract excludes them
while preserving exact aliases. The original launch used `n` programs for blocks
of four; the repaired launch uses the required ceiling. Signed-int32 extent and
index boundary witnesses remain in the original arithmetic helpers and tests.
Actual CPU unaligned FP32 views motivate the four-byte guard; aligned offset
views, unusual contiguous singleton strides, and empty spans remain supported.

## Public proofs and numerical scope

- `Correct.sin_kernel_correctness X O n` proves complete per-program Real sine
  results, termination, active access safety, masked tails and the exterior
  flat-memory frame, using the valid existing sine kernel proofs.
- `FPEquiv.index_commuted_equiv X O n` proves successful complete executions and
  equality of all typed memory cells for the actual original/index-commuted
  sources. Generic region names also cover exact aliases.
- `Misc/FPResults.lean::FPResults.kernel_results` retains the same successful source pair together
  with the complete exterior typed-memory frame over a raw initial state. This
  operational helper stays outside the public reader entry.
- The canonical FP32 source has the same explicit load, sine and store in both
  variants. The proof changes the executed integer prefix, preserving arbitrary
  initial registers, memory and unspecified inactive values.

Both the rewrite domain and numerical assumption list are empty.
`#print_fp_assumptions index_commuted_equiv` reports **none**. There is no new
sine-negation variant or pending numerical transformation experiment. The old
`SineObligation`/reference transports remain optional hypothetical primitive
conformance results; they are not public goals or numerical experiment TODOs.
The mathematical Real theorem does not claim exact GPU sine or an error bound.
Compiler, allocator, CUDA runtime and launch composition implementations remain
outside the kernel proof boundary.

## Validation

Targeted public/dependency builds use `/tmp/veritile-lean-toolchain` and
`LEAN_NUM_THREADS=2`, with standard-base axiom audits.

`python bench/repairs/tritonbench_g/sin_computation/check_contract.py` checks
unchanged original Python kernel AST, the sole index-product candidate edit,
exact guard correspondence, and CPU storage execution of both variants. Each
passes 21 rejection and 14 admission cases. Cases include exact aliases,
disjoint views in shared backing, differing output shapes, output-prefix tails,
empty inputs, offsets, nonfinite values, inactive NaN loads, and preserved
exterior bytes. Original layout/overlap/overlaunch failures remain exercised.

CUDA launches are mocked. No GPU execution or new compiler experiment was
performed; CPU sine is an indexing/storage oracle, not GPU numerical evidence.

`Misc/SourceBinding.lean` retains the source-identity helpers under their existing `FPEquiv` names.
