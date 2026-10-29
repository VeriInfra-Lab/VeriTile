# layer_norm_welfold: kernel-only two-pass layer normalization

The canonical candidate is `layer_norm_welfold_repaired.py`, selected by
`task_progress.json`. The original Python remains unchanged. Despite the
filename, the source uses two ordinary sum passes, not Welford triples.

## Independent mathematics and precision

For original logical FP32 input values `X[S,D]`, `W[D]`, and `bias[D]`:

```
mean[r] = sum_j X[r,j] / D
variance[r] = sum_j (X[r,j] - mean[r])**2 / D
rstd[r] = 1 / sqrt(variance[r] + 1e-5)
Y[r,j] = ((X[r,j] - mean[r]) * rstd[r]) * W[j] + bias[j]
```

All sums above range over `j < D`; `D > 0`. These formulas live in
`Contract.lean` and do not refer to source execution, chunking or reduction
order. Variance is nonnegative and epsilon makes the Real inverse positive.

Input storage, accumulators, means, rstd, centered squares and affine
operations are FP32. **Y storage is BF16.** The source retains both FP32 casts
at each input load, increasing chunk order, the chosen binary reduction tree,
FP32 denominator conversion and epsilon, division, `libdevice.rsqrt`, and BF16 conversion at
the final store. No arithmetic is moved between passes. Both configurations
remain `XBLOCK=1`, `RBLOCK=1024` or `2048`, eight warps, one stage.

`TensorBridge.layer_norm_welfold_real` interprets this source over ideal Real
arithmetic and proves the independent formulas. The BF16 cell tag denotes
the destination type; Real erasure does not simulate BF16 rounding. No
FP-to-Real equality or error bound is claimed.

## Checked domain and actual Tensor loads

The existing ordered Python assertions are unchanged:

- Strided FP32 tensors on CUDA device 0, `X[S,D]`, `W[D]`, `bias[D]`,
  with `S >= 0` and `D > 0`.
- Exact strides `(D,1)`, `(1,)`, `(1,)`, zero storage offsets, and no
  unresolved negative view bits. No copies or normalization occur.
- `D + 2048 < 2**31` and `S*D + 2048 < 2**31`, preserving the source signed
  int32 arithmetic for both reduction widths, padded lanes and loop exit.
- Nonempty pointers divisible by 16, retaining the selected historical
  native profile's alignment. Empty X pointers are not queried.
- All FP32 words admitted. Read-only inputs may alias or overlap.

`TensorBridge.python_checks_iff_preconditions` in `Misc/TensorBridge.lean` proves
exact agreement between the ordered
checks and `Admission.Contract`. `Proofs/InputBinding.lean` binds those same
metadata observations to the original `Torch.FP32.Tensor` objects. Ordinary
Tensor API laws interpret the observed shape, strides, storage offset and
pointer; they assume no kernel output or execution. The public theorems
construct the initial regional memory from one shared physical reader at
`data_ptr + 4*i`. They derive logical-value equality and active pointer bounds
inside the complete output proof. No extra `is_contiguous` guard is required.

The output arguments denote caller-provided BF16 Y and FP32 mean/rstd regions.
They must be distinct from each other and from the input regions. The kernel
proof covers all cells outside the active Y/mean/rstd stores, including unused
cells of those regions. The physical input observations and this complete
regional memory frame are part of both public results.
`TensorBridge.layer_norm_welfold_native_physical` additionally uses little-endian `read32`
on the original byte heap and actual `BitVec 64` output pointers. Ordinary
output-buffer capacity, alignment, disjointness and freshness facts give exact
four-byte mean/rstd stores, two-byte BF16 Y stores, and preservation of every
other byte, including all input backing storage. `Physical.store_bytes` proves
the regional store-to-byte interpretation. The BF16 conversion port supplies
the stored BF16 encoding in the low 16 bits of the shared word carrier.
Allocation, output Tensor construction, stream handling, autotune trials and
wrapper return execution are outside this kernel-only theorem.

## Source and implementation proofs

`Kernels.lean` contains two complete readable ordinary `triton` programs and
has no proof imports. Each quotation is the complete original Python function
body, including the source names, `range(0, rnumel, RBLOCK)`, casts, two-dimensional
tiles and `None` masks. Caller pointer/scalar types, XBLOCK=1, the RBLOCK alias
and the selected binary sum tree are configured outside the body.
`SourceCode.rowKernel` uses `layer_norm_welfold_repaired.py` unchanged. The explicit
`layer_norm_welfold_row_offset_reused.py` candidate expresses the existing Lean
row-offset reuse: `SourceCode.rowOffsetReusedKernel` computes `rnumel*x0` once
per normalization chunk and reuses it for the X load and BF16 Y store. This
removes one repeated integer multiplication; no measured speedup is claimed.
Both sources retain all three loops, casts, masks, barriers, arithmetic order,
ordered reduction trees, and stores. `SourceCode.launch` runs exactly S row
programs; S=0 runs the empty grid.

The unrolled proof reference is retained in `Proofs/LegacySource.lean`. Its
primitive ports follow the actual device-library rsqrt and assigned FP32
epsilon. It is not a claim that the former `tl.rsqrt` model or an untyped
epsilon literal is FP-equivalent. `check_pair_source.py` verifies the whole
Python module and requires the row product definition before both uses;
`jit-correspondence.json` checks both raw bodies and their invocation metadata. `Proofs/RawAccumulate.lean` and
`Proofs/RawStatistics.lean` connect the actual loop iterations to the
original ordered recurrences. `Proofs/RawNormalize.lean`, `RawRow.lean`
and `RawGrid.lean` prove every output and the full exterior frame.
`ReadableBoundary.preserve` proves equality of complete final memory from the
corrected unrolled reference to each actual source. It does not assume source
identity or a numerical equality to the former primitive ports. `RawSafety.lean`
separately follows the actual range loops and
proves all active accesses stay inside the original IO windows.

The `PairIO` namespace in `IO.lean` exposes the complete actual source schedules with the original
FP32 input/statistic channels and BF16 Y channel. `Correct.kernel_real` now
uses that readable original source and still states the independent mean,
population variance, inverse and affine formula. The public
`FPEquiv.row_offset_reused_equiv` compares the original and row-offset candidate
as `original ≡[[]] rowOffsetReused`, for arbitrary carrier words and algebra,
each retained Configuration and original Separation. The judgment includes
both successful full-grid runs, all three outputs, access safety and both
complete exterior frames. No finiteness guard or floating numerical law is added.

The optional Tensor/native/physical results below use the corrected unrolled
reference and retain their explicit modeling boundaries. They remain auxiliary;
the new main source pair and Real specification do not depend on `Misc/`.

`TensorBridge.layer_norm_welfold_native` runs the typed source over explicit
operation graphs and observes the successful result with a native instruction
profile. The independent reference uses separately defined expression
constructors, lane recurrences and binary trees in `Proofs/SourceGraph.lean`.
Its atoms contain only original input columns, D, and epsilon. It contains no
expected statistic, output oracle, failed-run comparison or source-execution
result. `Proofs/KernelGraph.lean` proves only active logical input columns are
used. `Proofs/NativeInputs.lean` attaches each leaf's physical address and raw
FP32 word to the original logical Tensor element.

## Native and compiler boundary

`Proofs/NativeKernel.lean` makes the compiler profile explicit. It chooses
FMA contraction and operand order from expression shape, literal metadata and
physical leaf identity, without inspecting raw input words. At a fused site,
observation executes a genuine ternary `fmaRN` primitive. **There is no premise
that fused arithmetic equals separately rounded multiplication and addition.**
The selected default-fusion profile constrains adjacent FP32 mul/add sites;
addition and multiplication operand order remain explicit for NaN payloads.

Shape-only switches also record compiler elimination of known-zero additions,
division by literal one, and identity FP32 casts. These decisions are part of
the native source interpretation, not numerical rewrite equalities used to
repair a mismodeled source. The BF16 destination conversion remains explicit
and cannot be erased by this profile. Chunk order and the selected reduction
tree remain unchanged. A profile cannot substitute a new expression, a
reduction answer or an input-dependent output oracle.

The instruction ports specify literal materialization, FP32 scalar arithmetic,
device-library rsqrt/division behavior, denominator conversion, casts and BF16 conversion.
These and the compiler profile are **modeling boundaries**, not deferred
numerical experiments. The theorem proves complete graph correspondence;
it does not prove the compiler or hardware implementation of those ports.
It preserves every FP32 word, including NaNs, infinities and signed zero,
without a finiteness guard or any exact-Real law.

No arithmetic transformation law is required by the current public results,
and no numerical experiment is deferred for this kernel revision. No new GPU
conformance or native replay is claimed. The earlier `LoweringPlan` and
`LocalLaws` proof files remain historical proof reuse; their optional
split/fused numerical equality is not a premise of the current FP theorem.

## Repair, counterexamples and validation

The repair still changes only three kernel sites: add `other=0.0` to both
statistics loads and use `tl.where(rmask, tmp9 * tmp9, 0.0)` for the variance
contribution. Normalization is unchanged. For `[2,2,2]` at RBLOCK=1024,
unspecified padded loads contaminate the original mean. Fixing only those
loads still makes each padded variance lane contribute `(0-2)^2 = 4`.
The masked square fixes that second defect. These controls are not FP/Real
rounding counterexamples.

Run `python bench/repairs/tritonbench_g/layer_norm_welfold/check_contract.py`.
It verifies the exact source delta and guards, then tests 21 rejections,
17 admitted CPU source executions, both RBLOCKs, tails, aliases, S=0, D=1,
BF16 output and the two failing controls. CUDA/host operations in that checker
are mocked; they are regression evidence, not a formal host-scope extension
or a GPU numerical experiment. Lean checks include the public specifications, their full local import closure
and standard-base axiom audits.

The earlier wrapper-level result, historical review hash, and old numerical
classification are recorded separately in `HISTORY.md`. No historical
acceptance ledger or original source is changed by this kernel revision.

`Misc/NativeCorrect.lean`, `Misc/TensorBridge.lean` and other optional entries
are rechecked with the corrected device-library port and FP32 epsilon. The
selected Python repair and Contract are unchanged. Targeted Lean builds and
a joint import cover Correct, FPEquiv and the retained native/Tensor entries;
the final batch is also replayed from an empty project build directory.
Current validation evidence belongs in the tracked `bench/review-checkpoints/`
directory. Historical cache receipts are not inputs to the proof build.
No GPU, native replay or new numerical experiment is claimed for the integer
address reuse. The independent comparator and Python dependencies are reported
separately when the current host cannot run them.
