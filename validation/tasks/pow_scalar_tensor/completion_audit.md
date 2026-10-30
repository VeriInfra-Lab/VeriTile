# Verification audit

Current selected public entries: [Contract.lean](../../../bench/tritonbench_g/pow_scalar_tensor/Contract.lean), [Correct.lean](../../../bench/tritonbench_g/pow_scalar_tensor/Correct.lean), [FPEquiv.lean](../../../bench/tritonbench_g/pow_scalar_tensor/FPEquiv.lean). The [contract review](../../contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](../../../bench/tritonbench_g/pow_scalar_tensor/Correct.lean) and [FPEquiv.lean](../../../bench/tritonbench_g/pow_scalar_tensor/FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

The fp32/int32 exponent families with natural strides and six floating output
labels passed the dependency-aware public build and exhaustive theorem audit.
The full Python domain remains incomplete: signed strides, integer scalar
conversion and nonfloating output conversion are still separate source paths.
Independent comparator replay is unverified.

## Reproducible public gate

```sh
LEAN_NUM_THREADS=4 /tmp/veritile-lean-toolchain/bin/lake build \
  +bench.tritonbench_g.pow_scalar_tensor.Correct \
  +bench.tritonbench_g.pow_scalar_tensor.FPEquiv \
  +bench.tritonbench_g.pow_scalar_tensor.Proofs.Audit
```

Receipt: `/tmp/veritile-pow-integer-public-v2.log`, exit 0, **3463 jobs**.
The exhaustive audit checked **406 theorem declarations**, including
private/generated helpers. Only `propext`, `Classical.choice` and `Quot.sound`
appear. Both full typed equivalence specifications report **no FP assumptions**.
There are 18 explicit axiom checks and
4 explicit formula noncircularity checks.
Existing linter warnings remain; the final build has no compiler errors.

## Source and execution evidence

- Both admitted exponent families preserve both complete branches, independent
  strides, the unused cdiv, base fp32 conversion, destination conversion and
  unspecified padding. Exact typed AST checks cover all 48 templates.
- The int32 source retains its exponent carrier and the separate native powi
  callback. Its source projection preserves the signed32 validity guard.
- Four dimension-general finite-flat contracts cover one-tile and grid-stride
  execution for both exponent families, arbitrary N/B/trip counts and both
  variants. They include actual trace safety, successful execution, independent
  initial-input power values and preservation of every raw cell outside writes.
- Integer contracts use typed input arrays and real `zpow`; active signed32
  premises cover the actual values loaded. The load proof evaluates the masked
  oracle conversion to integer zero under explicit zero undef, so all computed
  lanes satisfy the real guard, including inactive lanes.
- Separate finite allocations discharge every contract premise. Program id 1,
  grid width 2, strides 2/3 and a partial final tile exercise two grid iterations.
  Floating exponents 2/3/1 yield 4/8/2; integer exponents -2/-3/-1 yield
  1/4, 1/8 and 1/2. The same integer flat run retains its full raw frame and
  integer sentinels of 37 at masked/skipped outputs.
- Complete typed numerical executions cover all 48 templates and retain output
  storage tags. They use actual real arithmetic and integer power with identity
  casts. They are not IEEE or GPU execution results.
- `IntegerExecution.range_failure` checks actual complete source failure for
  any out-of-range integer input in its nonempty launch. Both immediately
  invalid endpoints, 2147483648 and -2147483649, fail across all 24 integer
  templates rather than wrapping.
- `powSpec`, `IntegerSteps.powSpec` and both finite exponent functions pass
  explicit noncircularity checks against source/execution definitions.

## Compatibility and discovery

All **36 historical declaration headers and 33 docstrings** match
`0ad438f5^:bench/tritonbench_g/pow_scalar_tensor/PowScalarTensor.lean`.
Receipt: `/tmp/pow_scalar_tensor_final_history_audit.json`.
The historical two specifications retain their original scope; new proofs do
not reinterpret those declarations as full typed Python coverage.

The port root has only `Kernels.lean`, `Correct.lean` and `FPEquiv.lean`.
Public imports reach 27 implementation modules; `Proofs/Audit.lean` is the
sole separate entry and reaches all 28 local modules. There are no orphans.
Receipt: `/tmp/pow_scalar_tensor_import_closure.json`.

The eight public records and Python correspondence are recorded in
`/tmp/pow_scalar_tensor_public_records.json`. All target
`pow_func_scalar_tensor_kernel_rank_1` at Python line 128, including the
inlined helper `pow_func_scalar_tensor` at line 124.
Generated review sheets are in `/tmp/pow_scalar_tensor_spec_sheets/`; both
have no self-reference flag and no missing-summary flag. Coverage registry
integration is handled separately from this directory's proof gate.

The pinned frontend dispatch script was rerun against the recorded Triton
3.5.1 snapshots. It selected `__nv_powf` for fp32 and `__nv_powif` for int32,
rejecting the thirteen other checked labels. Receipt:
`/tmp/veritile-pow-pinned-dispatch-current.json`.
This checks production overload selection only, not native lowering or GPU
execution. `source_scope.md` records native-width, signed-address, conversion,
IEEE, scheduling and replay boundaries.
