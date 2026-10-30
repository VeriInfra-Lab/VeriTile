# Local verification audit

Current selected public entries: [Contract.lean](../../../bench/tritonbench_g/rmsnorm_fused_llama/Contract.lean), [Correct.lean](../../../bench/tritonbench_g/rmsnorm_fused_llama/Correct.lean), [FPEquiv.lean](../../../bench/tritonbench_g/rmsnorm_fused_llama/FPEquiv.lean). The [contract review](../../contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](../../../bench/tritonbench_g/rmsnorm_fused_llama/Correct.lean) and [FPEquiv.lean](../../../bench/tritonbench_g/rmsnorm_fused_llama/FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

Status: the dependency-aware public build and exhaustive theorem audit passed
on the current shared DSL and int32 semantics. Coverage registry integration
and independent comparator replay remain separate gates.

The completed command was:

```sh
LEAN_NUM_THREADS=4 /tmp/veritile-lean-toolchain/bin/lake build \
  +bench.tritonbench_g.rmsnorm_fused_llama.Correct \
  +bench.tritonbench_g.rmsnorm_fused_llama.FPEquiv \
  +bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Audit
```

Receipt: `/tmp/rmsnorm_fused_llama_public_build.log`, exit 0, **3457 jobs**.
This supersedes the old leaf receipts from another environment. Existing
linter warnings remain; the completed build reports no compiler errors.

- The complete two-loop source and exact memory/cast/precision AST checks cover
  all 216 independent X/W/Y floating profiles and both variants.
- Independent logical-row mathematics, source decomposition, complete successful
  execution, trace safety, candidate transport and full flat IO/frame passed.
  The contracts allow arbitrary N and positive B; the source variance loop
  contains no additional `where`.
- Whole typed FP equality passed with **no FP assumptions**, including complete
  states, aliases, undefined values and failures.
- Eight typed numerical single/multiblock witnesses passed with actual real
  arithmetic, sum and square root, identity casts and stored values. They cover
  both variants, fp64 input/weight paths, N=1 and N=3 with B=2; rstd=1/3,
  active outputs=2 and adjacent/tail sentinels=7. Unsupported backend operations
  return `none`; the power operation uses actual real power.
- The finite erased-real witness discharges every IO and allocation premise for
  all profiles and both variants. With X/W/Y extents 9/4/10, row=1, stride=5,
  N=3, B=2 and eps=5, its same successful flattened execution returns three
  values of 2 and retains the full raw frame. Adjacent/tail Y and masked W
  sentinels retain their integer dtype and value 37.
- All five public specification heads and the finite witness passed explicit
  axiom checks. The exhaustive audit checked **852 theorem declarations**,
  including private/generated helpers, using only `propext`, `Classical.choice`
  and `Quot.sound`.
- `General.rowSpec` passed the noncircularity check against `originalTyped`,
  `typedKernel`, `VeriTile.Triton.exec` and `VeriTile.Triton.execR`.
- All **33 historical declaration headers and 26 docstrings** compared equal
  with `0ad438f5^:bench/tritonbench_g/rmsnorm_fused_llama/RmsnormFusedLlama.lean`.
  Receipt: `/tmp/rmsnorm_fused_llama_final_history_audit.json`.
- `Correct` and `FPEquiv` reach 18 of the 19 local modules; `Proofs.Audit` is the
  sole separate entry and reaches all 19. No proof module is orphaned.
  Receipt: `/tmp/rmsnorm_fused_llama_import_closure.json`.

The five public records are `Correct.rms_norm_fwd_fused_llama_output_summary`,
`Correct.rms_norm_fwd_fused_llama_io_correctness`, `Correct.original_real_correct`,
`Correct.candidate_real_correct` and `FPEquiv.source_equivalence`, where the
prefix denotes the physical entry file. All correspond to Python
`_rms_norm_fwd_fused` at line 7. The two historical single-block declarations
retain their old scope; the new real contracts cover the complete logical row.

See `source_scope.md` for the floating family, native admission/width, signed
addressing, integer/mixed conversion, allocation, initial snapshot, IEEE and
per-program boundaries. These results do not establish the full Python domain
or independent comparator replay.
