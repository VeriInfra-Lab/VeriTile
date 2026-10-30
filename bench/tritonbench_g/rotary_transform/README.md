# rotary_transform

- Source file: `rotary_transform.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Concrete fp32 correctness

`FPCorrect.lean` now exposes `rotary_transform_fp32_correctness` for all three
already selected entry forms: the non-interleaved paired row, fixed-length
metadata row, and variable-length metadata row. `Proofs/FPRow.lean` restores all
four Python `.to(tl.float32)` register casts. `rotary_transform_source_projection`
proves correspondence with the complete existing selected source after dtype
erasure, including both stores and the metadata branch. The source evidence is
`rotary_transform.py:38–52` (paired arithmetic and stores), with metadata at
lines 18–36. This profile is fp32 storage and fp32 registers; the existing tests
use `torch.randn` without an overridden dtype (the normal fp32 default).

The carrier is `FP.Value .fp32`. `FP.Software.scalarFP32` invokes the concrete
software `FP.mul`, `FP.sub`, and `FP.add` at every source operation. With inputs
read independently from the initial memory snapshot, both active output halves
are proved equal to `sub(mul(x0, cos), mul(x1, sin))` and
`add(mul(x0, sin), mul(x1, cos))`, respectively. Each operation rounds according
to `cfg`; the formulas contain no abstract algebra callbacks. Masked cache loads
retain the source's default cosine 1 and sine 0. Successful execution and every
cell outside the union of the two active write sets are proved as well.

The sole readback gate is the original nonzero output component stride. No
finite-value, exact-rounding, reassociation, or output-value assumption is added.
Input/output aliasing is allowed because both input halves are loaded before
either store. The proof also allows tile padding wider than the rotary half
dimension: disjointness uses the active-lane masks directly.

`rotary_transform_bit_success` covers all three entry forms, with empty initial
registers, one active pair, a masked tail, and a different initial destination.
Both `1 - 2^-25` and `1 + 2^-25` become bit pattern `0x3f800000`; their exact
rational results differ from 1. The witness therefore exercises real rounding.

The existing full 2D/interleaved/offset-tensor/conjugate lowering assertions are
still lowering assertions, not extra value coverage. In particular, the shared
DSL currently lowers unary minus as `0 - x`, which differs from a bit sign flip
on positive zero; the conjugate path requires a correct negation primitive or a
separate source refinement. CUDA/compiler equivalence (including contraction,
FTZ, NaN policy, and machine-address arithmetic) remains a separate refinement
boundary; this theorem is a concrete, separately rounded software fp32 contract.

Validation: directed Lean compilation of all four `Proofs/` helpers and the
public entry passed, including `#axiomsClean`; logs are
`/tmp/rotary-transform-{row-initial,contract,shape,software,public}.log`.

The separate [checked FP32 revision](../../repairs/tritonbench_g/rotary_transform/README.md) preserves original preparation and return behavior with an explicit CU and pointer contract.
