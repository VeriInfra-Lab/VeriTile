# Primary acceptance versions: Real / FP32 / BF16

The user approved this scope on 2026-10-16: **183 original programs, with
Real / FP32 / BF16 as the three primary versions, plus types indispensable to
the original program**. This replaces exhaustive caller-dtype expansion as the
default development target. All source-declared production stages and branches
within each selected profile remain in scope. Exactly registered numerical
conformance assumptions may be validated later.

## What is maintained

| Version | Required interpretation |
| --- | --- |
| Real | A reviewed mathematical abstraction of the source algorithm, with ideal arithmetic and an explicit domain. Real is not a native Torch/Triton dtype. |
| FP32 | The actual selected source profile, including its input/output storage, intermediate arithmetic, casts, accumulation and target configuration. |
| BF16 | The actual selected BF16 source profile. FP32 accumulation or output is retained when the source uses it; this does not require every port to be BF16. |

Maintain shared source translations, control-flow, indexing and memory proofs
once. Reuse a generic proof by recording the relevant version instantiation;
prove differences only where precision or source behavior changes. Existing
accepted shared obligations need an applicability review, not duplicate proofs.
The three versions do not require three copied implementations or proof trees.

Keep source-required integer/index/mask types, packed formats, quantization
outputs, fixed FP16 paths and mixed precision. Record the source reason for each
additional profile. Do not expand arbitrary runtime dtype families or Cartesian
input/output combinations by default. Keep completed wider proofs and their
numerical registrations as reusable evidence.

## Completion and exceptions

Every task records Real, FP32 and BF16 separately. Each record identifies its
source or explicitly named variant, mathematical/runtime domain, proof scope,
remaining gaps, and numerical assumptions. Stage/branch and memory obligations
are shared where justified; storage widths, casts and precision-dependent paths
retain their actual interpretation. Known source bugs remain explicit.

A mathematical theorem alone does not establish floating execution correctness.
A successful build, a profile name, or the presence of BF16 output does not
establish full BF16 coverage. A known source rejection is a support limitation,
even when that rejection is proved correctly. A repaired BF16 implementation is
a separate variant with its own source/proof correspondence. Variant completion
does not increment the original-source FP32/BF16 completion count. Support and
native observations stay tied to their reviewed target/configuration domain;
a single compile failure does not establish failure of every configuration.

For a source that mandates another type, retain that required profile and record
how it relates to the three requested versions. Do not silently relabel FP16 as
FP32/BF16 or count a missing/unsupported version as complete. Report any reviewed
applicability exception separately. A program is counted as having all three
versions complete only after all three have reviewed complete scopes. A reviewed
required-type exception is reported separately as a policy-resolved exception;
it never means that the missing FP32/BF16 implementations are complete.

Unverified numerical assumptions must retain exact operation, declaration,
dtype/configuration and callback bindings. Missing source translations, host
composition, branch coverage, memory/frame proofs and known false statements
cannot be deferred as numerical assumptions. Build/trust checks, native
execution and official-comparator results remain separately reported.

## Migration and progress

The historical schema2 ledger and its checker keep their original meaning:
whole-original source scope under the earlier dtype policy. They are retained as
evidence; their accepted count is not automatically the three-version count.
The historical policy remains in `bench/original_program_acceptance.md` so existing
reviews keep their exact input bindings.

The first per-version inventory is an evidence map, not a new acceptance run.
It does not revoke existing proofs or reset their progress. New-policy totals
remain **not yet audited** until the source-specific mappings are reviewed.
A dedicated per-version acceptance gate has not yet been installed.
Use explicit version records and exact existing numerical checks for subsequent
acceptance; do not weaken the historical checker or add Real to runtime DTYPES.

See `bench/current_source_acceptance.md` for the latest committed counts and evidence
snapshot. Future implementation work follows this three-version scope.
