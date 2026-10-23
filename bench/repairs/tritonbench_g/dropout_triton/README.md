# dropout_triton: given-mask kernel correctness

The caller supplies an int32 keep mask; this kernel generates no random values.
Every nonzero keep value, including `2` and `-1`, is true. The independent real
formula is `x[i] / (1-p)` on kept coordinates and zero on dropped coordinates.
A kept coordinate at `p=1` has no conventional real value; this mathematical
singularity is exposed explicitly below. An all-zero mask at `p=1` produces
zero, preserving the benchmark's original endpoint test.

## Source, precision and candidate

`Kernels.lean` contains the readable canonical Triton DSL, selected at FP32 data
and output with signed-int32 keep input. Both public proof entries use this
same source. The existing candidate changes only integer index multiplication:
`pid * BLOCK_SIZE` becomes `BLOCK_SIZE * pid`. It retains all floating
expressions, masks, loads and stores. Both versions appear in
`dropout_triton_repaired.py`; the original `_dropout` AST is unchanged.

The ordered floating expression converts the runtime Python float `p` to FP32,
computes FP32 `1-p`, evaluates `x/(1-p)` eagerly, selects that quotient or zero
using integer nonzero truthiness, then stores FP32. The DSL's explicit
same-dtype store cast represents Triton's destination conversion. Division is
the selected primitive's ordinary Triton FP32 division, not a newly asserted
correctly rounded IEEE division. Basic literal conversion, arithmetic, casts
and load/store meaning are modeling boundaries; they are not numerical
transformation experiments.

The candidate does not change floating-point arithmetic. Its equivalence needs
**no numerical assumptions**, no finite-value conditions, and no restrictions
on inactive lanes. It remains valid for nonfinite FP32 data, zero denominators,
nonbinary masks and arbitrary inactive load values.

## Input contract and memory meaning

The Python assertions match `Contract.lean` and
`TensorBridge.python_checks_iff_preconditions`:

- Strided CUDA Tensors on the same device, with exactly the same shape:
  FP32 `x` and signed-int32 `x_keep`.
- Default-contiguous layout and no unresolved negative-view bits on either
  input. Contiguous offset views, singleton dimensions and scalars are allowed.
- `numel < 2**31`; for nonempty inputs, both actual data pointers are divisible
  by four. Empty inputs impose no unused pointer requirement.
- `p` is a builtin Python `float` in `[0,1]`. Integers, bools, NaN, infinities
  and out-of-range values are rejected. The scalar is passed unchanged.

NaN/Inf data and arbitrary int32 masks remain admitted. There is no mask
sampling, binary-mask guard, denominator guard, input copy or layout adapter.
The two read-only input storages may overlap. The wrapper's output allocation
and device context retain their existing behavior and are not new proof tasks.

`Proofs/MemoryContract.lean` connects these metadata conditions to addresses
and logical values. FP32 data uses the shared `Torch.FP32.TensorLaws`. The
int32 binding reuses the common four-byte storage geometry, with separate
int32 dtype/layout, physical-index, actual-pointer and contiguity laws.
`direct_keep_read` relates the signed raw word at `data_ptr + 4*i` to the
logical keep value when the negative-view flag is absent; `keep_active_bytes`
proves alignment and bounds, including nonzero storage offsets. These input
laws contain no kernel-output premise and no allocator model.

The shared [Tensor API binding](../../../../VeriTile/Torch/FP32.lean) cites
`data_ptr`, `storage_offset`, `is_contiguous` and `resolve_neg`. The
[physical load/store model](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts)
records byte-address, alignment and mask semantics. A scalar FP32 or int32
access occupies four naturally aligned bytes; this is not a claim about the
alignment required by arbitrary vectorized machine instructions.
`native_indices` reuses the existing finite-index proof for every launched
lane, including inactive tails.

## Public proof scope

The primary `Correct.dropout_kernel_correctness` specification and the ordinary
candidate theorem `Correct.dropout_index_commuted_correctness` in
`Misc/AuxiliaryCorrect.lean` use the shared `MixedMaskedKernelIO₂` regional IO judgment.
`dropoutIO` visibly binds the exact FP32/int32 source, typed channels, three
address windows and independent masks. The logical formula consumes arrays,
while the shared relation provides actual source execution and the frame
outside active output coordinates. They impose no input separation, `FlatAlloc.Disjoint`, or
zero-valued inactive-load condition. Actual source access safety is now
connected inside the shared IO judgment, using the existing load/store proof
and a proof that the candidate's pure index prefix preserves that trace. `TensorBridge.checked_tensor_reads` uses one physical
32-bit-word reader for both Tensor inputs, so identical storage and partially
overlapping heterogeneous views remain admitted. It interprets each shared
word as FP32 data or signed-int32 keep according to the corresponding input.

The earlier flat-memory results remain explicitly named
`RealEndpoints.dropout_kernel_flat_correctness` and
`RealEndpoints.dropout_index_commuted_flat_correctness` in
`Proofs/RealEndpoints.lean`. Their pairwise-disjoint `FlatAlloc`
premise is stronger than Python admission; they are optional corollaries and
do not replace the primary alias-permitting result. `Proofs/Region.lean`
reuses the original regional execution and index-prefix proofs through exact
canonical source projections, without introducing wrapper or allocator work.

The total Real projection uses Lean's total division. It is named separately
from the conventional partial formula `mathematicalValue`. That independent formula
and its `MathematicalDomain` are defined in `Contract.lean`. The theorem
`RealEndpoints.dropout_mathematical_correct` states its additional mathematical domain
explicitly: every kept coordinate must have `p != 1`. This is not a Python
guard. The full typed source theorem still covers all admitted inputs,
including kept coordinates at `p=1`, using the backend's division semantics.
A Python `p<1` does not imply a nonzero FP32 denominator: `1-2**-26` rounds to
FP32 one. No theorem conflates that machine result with the real quotient.

`FPEquiv.dropout_index_commuted_equiv` proves that both actual typed sources
execute successfully and agree on every memory cell. It reuses the existing
integer rewrite theorem and adds unrestricted successful execution, so it does
not merely equate two potentially failed runs. `dropout_kernel_result` exposes
the ordered source-precision output formula under the standard interpretation
of same-dtype casts. `dropout_kernel_frame` proves exterior memory preservation
without numerical or finite-operand premises. These two operational helpers
retain their names in `Misc/FPFacts.lean`, keeping the public FP entry focused
on the shared floating-program equivalence.

`Kernels.lean` is source-only, `Contract.lean` contains inputs and the independent
projection formula, public correctness/equivalence live in their respective
entry files, required proof helpers stay in `Proofs/`, and optional entries live
in `Misc/`. Old grid, wrapper,
allocator and physical-wrapper results remain optional helpers. Reused typed
interpreter helpers share existing grid utility imports, but the public
specifications concern one kernel and add no wrapper execution or launch
composition obligations. No public result requires returned Tensor identity
or allocator behavior. `Proofs/ScalarABI.lean` retains the earlier
formal signed-zero conversion/denominator lemma as optional modeling evidence;
it adds no native-conformance or experimental premise.

## Original-source limitations

The original benchmark remains unchanged. It checks only x's contiguity.
For keep backing storage `[1,0,1,0]` viewed with stride two, the logical keep
values are `[1,1]` but pointer-linear loads read `[1,0]`. With `x=[2,4]` and
`p=.5`, the expected supplied-mask result is `[4,8]`, while the second output is
zero. A too-short keep buffer instead permits an out-of-bounds read.

A contiguous unresolved negative FP32 view similarly has logical values that
differ from the raw loaded words. Contiguity alone also does not guarantee
natural pointer alignment. The revised public contract states these supported
conditions explicitly, rather than claiming correctness for unrestricted
original inputs. The preserved signed-index counterexample has
`pid=2**21`, lane zero and `n=2**31+1`: the offset wraps to `-2**31`. This is
separate from the postponed shared grid-product overflow issue.

## Validation

```sh
lake build bench.repairs.tritonbench_g.dropout_triton.Correct \
  bench.repairs.tritonbench_g.dropout_triton.FPEquiv
python bench/repairs/tritonbench_g/dropout_triton/check_contract.py
```

The checks verify unchanged original AST, an index-only candidate change,
exact guard binding, 25 rejected calls and 19 admitted CPU source executions.
Cases include empty/scalar/tail inputs, offset, identical-storage and partially overlapping heterogeneous inputs, nonbinary
masks, p endpoints, near-one conversion, signed-zero p, NaN/Inf data and all
three unaligned pointer remainders. CUDA allocation and launches are mocked.
Lean audits use the standard base and report no FP transformation assumptions.
No numerical transformation TODO remains. GPU execution and independent
numerical replay were not performed.

`Misc/TensorBridge.lean` contains the optional Tensor/check correspondence theorems;
`Correct.lean` and `FPEquiv.lean` retain the primary kernel specifications.
