# reversed_cumsum_scalar: kernel correctness and index commutation

The independent mathematical result is
`out[b,h,t] = sum(input[b,h,k], t <= k < T)`.
The required proof scope is one kernel program, following the approved
`add_value` standard. The wrapper supplies checked input assertions; allocation,
autotuning and returned-Tensor orchestration are optional additional results.

`Misc/TensorBridge.lean` provides the optional Python-check correspondence and native Tensor program certificate. The main mathematical and FP declarations remain in `Correct.lean` and `FPEquiv.lean`; neither imports the bridge.

## Supported inputs and precision

The wrapper admits rank-three strided, contiguous CUDA FP32 Tensors, with no
unresolved negative view bit. The requested output dtype is None or FP32.
Nonempty inputs need a four-byte-aligned data pointer and `B*H < 2**31`.
Empty axes are supported without checking an unused pointer or row grid.
Contiguous offset views and unusual singleton strides are admitted. NaN/Inf,
signed zero and subnormal inputs are not rejected. There is no divisibility,
finite-value or `numel < 2**31` restriction.

[Contract.lean](Contract.lean) names exactly those assertions.
`TensorBridge.python_checks_iff_preconditions` proves their dynamic admission
equivalence, including non-Tensor rejection. `Proofs/KernelContract.lean` also
binds shape, FP32 storage, original data pointer and active-coordinate strides
to the kernel arguments. It derives output-address uniqueness and valid native
program IDs from ordinary Tensor metadata; it does not assume any output value.

The selected input, output and arithmetic precision is FP32. The five original
`(BT, num_warps)` choices remain `(16,2)`, `(32,4)`, `(32,2)`, `(64,8)` and
`(64,4)`. Explicit FP32 load conversion, configured reduction, descending
carry addition, forward inclusive scan, separate subtraction/addition, and
output conversion retain their source order.

## Source pair and mathematical proof

[Kernels.lean](Kernels.lean) contains the authoritative readable typed DSL for
both complete kernels. [Contract.lean](Contract.lean) contains the independent
ordinary suffix formula `mathematical`. `Kernels.source` is the production repaired kernel;
`Kernels.indexCommuted` commutes the two natural row/stride products in its
input pointer. All floating expressions, masks, stores and output pointers
are unchanged. The matching Python JIT candidate is
`chunk_global_reversed_cumsum_scalar_kernel_index_commuted` in
[reversed_cumsum_scalar_repaired.py](reversed_cumsum_scalar_repaired.py).
The checked wrapper continues to select the production kernel.

Both [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean) use exactly
`Kernels.source h c`. There is no replacement identity-equivalence source.
Only the Real interpretation projects dtypes away. Native signed-i64 casts and
arithmetic are represented by mathematical indices in this DSL;
`TensorBridge.native_programs` connects them to the complete fixed-width control,
clamped inactive coordinates, input/output offsets and byte-pointer spans.

`Correctness.reversed_cumsum_scalar_correctness` states `Spec.Real` with
shared `AlgorithmProgramIO.Implements`. `IO.lean` exposes one input row, one
output row, their original strides, no private scratch, and the complete
source loop launched at the selected row. `Interface.reference` is the
independent finite suffix sum. `Proofs/SourceSafety` proves the actual
masked pointer loads/stores safe throughout that same reverse loop;
`Proofs/IOComplete` reuses existing execution and mathematical results.
Every initial register and unspecified-value oracle remains admitted.

The unchanged `Correctness.kernel_correct` and `source_execution` statements
are ordinary theorems in `Proofs/KernelCompatibility.lean`. The physical
corollary retains `Storage.Placement`, disjoint FP32 allocation windows,
native address-space limits, every active suffix value, every input backing
cell, and every exterior cell. Cells are keyed by byte addresses; the theorem
has not changed into a raw-byte encoding claim. Both levels use the same
existing source execution proof. Neither a successful run nor desired output
is an input premise.

## Actual FP equivalence

`Equivalence.reversed_cumsum_scalar_equiv` compares the production kernel with
the displayed index-commuted candidate using profiled FP32 execution. It proves
equality of complete typed states and failures for arbitrary initial memory,
registers and arithmetic backends. Floating operations and precision tags stay
intact. The proof uses only natural multiplication commutativity; the numerical
assumption list and rewrite-domain guards are empty. The assumption audit
reports **none**.

`Equivalence.successful_pair`, retained in `Proofs/FPCompatibility.lean`,
constructs successful complete runs
for both sources on any admitted row and total configured scan backend, and
retains the exact source-precision output recurrence and memory frame. Thus
state equivalence is accompanied by a successful-execution theorem. The
backend interprets the identical operations on both sides; no native-to-IEEE
primitive-conformance premise or deferred numerical experiment is required.

This is not an equality between FP execution and the Real suffix formula.
For `[1e20, 1]`, the retained subtraction/scan expression gives `0` at the last
lane in the CPU FP32 interpreter, whereas the mathematical suffix is `1`.
No cancellation, associativity, reduction reassociation or error bound is
claimed. Compiler and runtime correctness remain external interpretation
boundaries, rather than task-level numerical TODOs.

## Original defects and repair boundaries

The unchanged original source has an unpadded boundary load. Its existing
complete Real proof gives the desired suffix plus the unspecified padded-tail
sum. At `T=17`, `BT=16`, input `2` and tail value `7`, first/last outputs are
`139` and `107`, instead of `34` and `2`. Explicit zero-padded masked loads in
the repaired production source remove that defect; an assumption about the
original load does not.

The original flat row addressing also disagrees with noncontiguous logical
views. The checked wrapper rejects those views and passes actual strides for
admitted contiguous inputs. The old i32 row product can wrap: row `2**30`
times `T=4` becomes zero instead of `2**32`. The repaired source widens indices
before multiplication and clamps inactive coordinates. Existing counterexample
and native-index proofs are retained.

The former public wrapper results remain in
[Misc/OptionalHostCorrect.lean](Misc/OptionalHostCorrect.lean) and
[Misc/OptionalHostFPEquiv.lean](Misc/OptionalHostFPEquiv.lean), under separate
optional namespaces. The public kernel entries do not import them or the local
host composition. The extracted input checks and memory bindings preserve
their existing proof bodies. Historical original files are unchanged.

## Validation

Targeted Lean 4.29.0 builds use `/tmp/veritile-lean-toolchain` with
`LEAN_NUM_THREADS=2`. Public kernel correctness, intended FP equivalence,
source identities and success certificates pass standard-base axiom audits.
The independent formula audit excludes both kernel bodies. The public IO
format migration separately compiled both reader entries, the actual source
safety and IO composition, unchanged physical endpoints and TensorBridge.
That migration did not rerun CPU controls, GPU execution or numerical experiments.

```sh
python bench/repairs/tritonbench_g/reversed_cumsum_scalar/check_contract.py
python bench/repairs/tritonbench_g/reversed_cumsum_scalar/check_kernel_variants.py
```

The contract check passes 12 rejections, 10 admissions and 60 CPU Triton
interpreter controls across all configurations and chunk boundaries. The
variant check verifies that the Python bodies differ only in the intended
index products, runs 60 candidate suffix/tail controls, and compares 10 pairs
bitwise including exceptional values, offset backing storage and guard words.
The original padding/layout and repaired-source cancellation controls pass.
CUDA is unavailable; these controls do not claim GPU execution or IEEE
conformance. No numerical transformation experiment is deferred.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.

## Raw Python function bodies

`Kernels.lean` quotes both complete selected functions from the existing
repaired Python file, checked independently by `jit-correspondence.json`.
The Python original and its existing candidate are unchanged. All explicit
casts, masks, zero padding, statement order and descending loops are copied.
Pointer element types and constexpr arguments are outside the copied body;
independent input/output `FloatDType` parameters retain all six cases, while
the existing host wrapper still requires FP32 storage.

The signed row address uses the original quotient and remainder, with each input and output stride kept independently. The candidate retains its existing integer-product commutations. The
internal execution views are proved equal to the current quoted sources.
Signed program IDs are part of the actual state; proof-only observation
adapters do not add source statements. Nonnegative lane coordinates use the
existing mathematical index projection, with native width/storage obligations
kept in the existing native binding and index refinements. These integer
connections assume no floating-point identity.
