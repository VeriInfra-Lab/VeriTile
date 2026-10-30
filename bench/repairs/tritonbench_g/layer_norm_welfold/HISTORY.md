# Historical wrapper-level result

This records the earlier contract revision. Its descriptions of public entry
points and deferred primitive experiments are historical, not the current
kernel revision. The current scope and numerical classification are in README.md.

# layer_norm_welfold: two-pass FP32 statistics and BF16 output

This candidate repairs three tail sites and checks the selected input contract
before entering the original host code. The original benchmark Python is
unchanged. Despite the filename, this kernel computes two ordinary sums; it
does not combine Welford triples.

## Independent mathematical target and actual precision

For each row `r` and logical column `j < D`, independently of chunking or
reduction order:

```
mean[r] = sum_j X[r,j] / D
variance[r] = sum_j (X[r,j] - mean[r])**2 / D
rstd[r] = 1 / sqrt(variance[r] + 1e-5)
Y[r,j] = ((X[r,j] - mean[r]) * rstd[r]) * W[j] + bias[j]
```

`D > 0`. The selected input storage is FP32; all source loads, explicit casts,
accumulators, means, centered squares and affine operations retain FP32
precision. Mean and rstd storage are FP32. **The source allocates Y as BF16**, so
the final store retains its BF16 conversion. Both explicit `.to(tl.float32)`
casts on each input remain. Epsilon remains the original `1e-05` literal.

The source runs each chunk in increasing order, then performs the selected
binary sum tree. Default fusion remains enabled, including possible native
affine FMA contraction. The proof does not replace reductions with arbitrary
associativity, assume IEEE nearest-even division for Triton's default division,
or silently remove FP32/BF16 casts.

`Correct.layer_norm_welfold_real` proves the independent formulas for the
complete ideal Real interpretation, including all returned cells. Its BF16
cell tag records the destination type; ideal Real erasure does not simulate
BF16 rounding. `FPEquiv.layer_norm_welfold_native` separately composes the
complete checked source with its native scalar graph under explicit local
numerical laws. No FP-to-Real equality or error bound is claimed.

## Checked input and output contract

- Arguments are strided FP32 tensors on CUDA device 0: `X[S,D]`, `W[D]`, and
  `bias[D]`, with `S >= 0` and `D > 0`.
- Strides are exactly `(D,1)`, `(1,)`, `(1,)`; all storage offsets are zero.
  Unresolved negative views are rejected. No copies or view normalization run.
- `D + 2048 < 2**31` and `S*D + 2048 < 2**31`. These are the inherited source
  index ABI bounds for both RBLOCK choices, including rounded inactive columns
  and the final loop increment. They are not a runtime grid-product repair.
- Nonempty input data pointers are divisible by 16, preserving the selected
  historical native profile's alignment. This is a profile restriction, not a
  claim that every scalar FP32 instruction requires 16-byte alignment. Empty
  X pointers are not queried. W and bias are nonempty because `D > 0`.
- Input data words need not be finite. Read-only inputs may share storage.
  Numerical conclusions still require the listed local primitive laws at the
  operands reached by the selected compiled plan.

`Admission.checkInputs_iff` proves exact equivalence between successful ordered
Python checks and `Admission.Contract`, including returned `(S,D)`. The public
mathematical and native results use that same checked contract. Ordinary
`TensorBinding` facts attach metadata and FP32 storage cells to source tensor
objects; the Python checks do not pretend to inspect all data cells.

The original `_DeviceGuard(0)`, `set_device(0)`, raw stream query,
`empty_strided_cuda`, `reinterpret_tensor`, autotune call and return order are
preserved. Both configurations remain: `XBLOCK=1`, `RBLOCK=1024` or `2048`,
eight warps, one stage. The proof covers arbitrary successful autotune trials
followed by the selected launch. It does not assume that different reduction
profiles produce identical floating results. `S=0` keeps the actual empty grid
and returns its allocated descriptors.

The returned tuple is exactly `(Y, original X, mean, rstd_view)`:

| Tensor | dtype | shape | strides | ownership |
| --- | --- | --- | --- | --- |
| Y | BF16 | `(S,D)` | `(D,1)` | fresh output |
| X | FP32 | `(S,D)` | `(D,1)` | original input object |
| mean | FP32 | `(S,1)` | `(1,S)` | fresh mean storage |
| rstd backing | FP32 | `(S,1)` | `(1,S)` | fresh inverse storage |
| returned rstd view | FP32 | `(S,1)` | `(1,1)` | aliases inverse backing, offset zero |

Successful allocation preserves caller live regions and provides distinct
output/mean/inverse backing regions. Those ordinary allocator facts are explicit
public premises. `inverse_view_alias` proves the exact reinterpretation and
address equivalence, including singleton column dimensions.

## Repair and counterexamples

The kernel changes are exactly:

1. Add `other=0.0` to the first statistics load.
2. Add `other=0.0` to the second statistics load.
3. Replace the centered square with `tl.where(rmask, tmp9 * tmp9, 0.0)`.

The normalization loop is unchanged. For a constant row `[2,2,2]` and
`RBLOCK=1024`, unspecified padded loads can contaminate the mean. Zero-filled
loads alone are insufficient: each padded column then contributes
`(0-2)**2 = 4` to the variance. The repaired masked square contributes zero.
The CPU source checks reproduce both controls and verify the repaired rstd.

## Proof reuse and current boundary

The reused closure is the latest complete canonical/ABI integration reviewed at
`bench/review-checkpoints/2026-10-18/layer-norm-welfold-canonical001-repair-integration/repair-review.json`
in the historical checkout. Its reviewed candidate SHA256 was
`0867ee30ffb2dcd5990ab23f9a14c8692452c953c80e0c98942a517977052ae7`.
The current wrapper adds checks; that old hash is provenance, not a hash of
this new candidate. No evidence archive, acceptance ledger or historical
acceptance count is copied or changed.

- `Kernels.lean` contains independent source definitions and no proof imports.
  `Proofs/SourceBinding.lean` proves equality to the historical materialization,
  including each sum tree and normalization iteration.
- `Proofs/Wrapper.lean` executes those source-only kernels for every trial and
  selected row, preserves the required host allocations/reinterpretation, and
  proves the adapter to the retained complete host theorem. The public entries
  execute this guarded wrapper; they do not merely import disconnected lemmas.
- The retained accumulator, statistics, row, grid, trials, Real and host proofs
  cover every valid column and returned tensor, plus the exterior memory frame.
  The exact missing original fused-layernorm helper closure was restored
  byte-for-byte as dependencies; no original Python changed.
- `NativeABI` proves signed-i32 parameter/row/chunk/mask/exit exactness for both
  configurations and the 2-byte/4-byte relative address displacements. Actual
  storage base validity remains an ordinary Tensor/allocator binding; this is
  not a proof of PyTorch or device allocator internals.
- `NativeTree` retains kernel-checked leaf permutations for the observed 1024,
  2048, singleton-width and vectorized reduction trees.
- `SourceBindings` contains nine scalar operation bindings. `LocalLaws`
  contains ten kinds of local numerical/peephole obligation, scoped to reached
  steps of a value-erased `LoweringPlan`. The plan contains source atoms and
  local syntax transformations, not result arrays, whole-reduction equalities
  or an input-dependent output oracle. The complete native host result and
  caller frame are derived by composition.

The historical review covered ten saved SM80 graph instances. This revision
does not claim a universal compiler theorem, JSON certificate replay in Lean,
new native GPU execution, or universal validity of the deferred numerical laws.
Their compiler/device numerical validation remains in the shared later batch.

## Validation

All task Lean modules and the required twelve restored original helper modules
compiled with Lean 4.29. Public theorem audits use only the standard base; no
`sorry` or new axiom was introduced. Mathematical targets also pass the
non-circular specification audit.

Run `python bench/repairs/tritonbench_g/layer_norm_welfold/check_contract.py`.
It checks the exact three-site AST repair, unchanged required host operations,
guard/source bindings, 21 rejected calls before allocation, and 17 admitted CPU
source executions. Cases include both RBLOCKs, preceding autotune trials,
`S=0`, `D=1`, aligned and tail widths, input aliasing, original X identity,
output BF16 storage, exact mean/rstd strides and the inverse backing alias.
CUDA allocation/device/launch operations are mocked. CPU arithmetic checks
are regression evidence, not native numerical conformance.
