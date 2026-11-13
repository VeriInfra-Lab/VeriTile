# Documentation audit — 2026-09-17

Scope: the public website, its repository design-note sources, and the
README / tooling references linked by the site. The Lean implementation was
used as the authority for interfaces and modeled guarantees.

## Corrected

- Replaced nonexistent `ComputeCorrect.Realizes` references with the appropriate
  exact or rounding-model API; corrected the `.toRealizes_without_Rounding`
  bridge and the explicit `R` argument to `RefinesAt`.
- Distinguished universally quantified `ComputeRefine.Realizes` from fixed-model
  `Refines R` and `io ⊨[R] f`; documented partial correctness versus the
  termination/output/frame guarantees in the KernelIO contracts.
- Documented `round_real` and `round_idem`, abstract cast/store rounding, and
  the remaining IEEE-754 boundary.
- Updated the implemented flat-memory bridge and atomic slices; corrected
  relocated files, reduction helper names, and standalone helper ownership.
- Marked the GELU strategy as research and linked the current axiom whitelist.
- Added semantic-caveat and trust-audit pages in both languages.
- Replaced the old migration script with repeatable source synchronization,
  correct localized links, and a check mode.

## Verification

- All generated design-note copies match their sources (26 pages).
- Production site build succeeds (51 pages).
- Built internal links, anchor targets, and linked repository files resolve.
- Public API names extracted from English design notes, cookbook pages, and
  README are resolved by the pinned Lean toolchain.
- `bench/examples/VectorAdd/RealCorrect.lean` elaborates, including its axiom and statement
  audits. `tests/lean/TritonSmoke.lean` and `tests/lean/LoopInvariant.lean`
  elaborate too. These runs emit existing unused-simp-argument warnings.
- Site CI now checks source synchronization and links. The existing artifact
  gate also checks documented API names.

## Limits

This audit does not rerun the complete benchmark proof/trust suite or the
full artifact release gate. Schematic proof fragments with placeholders are
identified as templates rather than standalone compilable programs. API-name
resolution is not a semantic proof of the prose; changes to assumptions and
theorem meanings still need review. External issue resolution is left to the
linked upstream reports.

## Website language update — 2026-09-20

The public website now publishes English only. The Chinese site copies and
language navigation were removed, and source synchronization covers the 13
English design notes. The original audit counts above describe the earlier
bilingual build.

The updated production build contains 26 pages, including the error page.
All 1,369 internal and repository links resolve. Generated pages use English
and contain no Chinese text or language selector; Chinese routes and sitemap
entries are absent. Repository design-note sources remain the authority for
the English pages.

## Current-code documentation audit — 2026-11-13

This audit starts from core main `d0e0d8406d74e7f25c04a65d974dbdb147155ec5`.
It covers all 91 tracked Markdown/MDX documents, their 13 generated website
references, and the website components that publish proof/CI links. Current
statements were checked against the workflows, Lake targets, public proof
entries and source-authoring frontend. Historical plans, fixture evidence and
measurement reports retain their original time and source scope.

Corrections include:

- Core versus VeriTileBench ownership: core has 18 worked-example groups;
  the benchmark collection has 19, including TritonBenchVectorAddition.
- Actual core PR/main and nightly CI scope, separate benchmark audits, and
  the library comparator's manifest-target/dependency selection boundary.
- Strict VectorAdd FP32 certificates and finite-operand domains, separate
  experimental rule dependencies, and the two-section assumption report.
- Original Python function bodies with external call/parameter bindings;
  theorem specialization is not permission to modify the authoritative body.
- Current example/helper paths and commands that build example imports before
  re-elaboration; stale workflow links and section anchors were repaired.
- Current fixed-batch paper two-gates protocol versus historical replicate
  reports, preserving every checked-in measurement and admission table.

Verification:

- All 91 tracked documents pass local/current-repository source and workflow
  link checks; all 13 generated references match their sources.
- Lean resolves 122 documented public API references. Both README Lean blocks,
  the tactic guide's association example, and the translation guide's source
  block compile. The latter is also proved equal to the actual VectorAdd body.
- The core example script passes source correspondence and Real/FP builds for
  VectorAdd, RowWiseSum and LogExp. No kernel or theorem source was changed.
- Twenty-nine targeted regression tests pass: documentation links, spec-sheet
  extraction, API validation, example dependency layout, dependency-profile
  consistency, coverage records and homepage-record freshness. New checks cover
  missing same-repository workflow/source links while leaving pinned history
  external. The built-site checker uses the same URL resolver.
- Production site build succeeds with 28 pages; 2,038 internal/repository links
  resolve. The existing VectorAdd recorded proof fingerprint remains current.

Limits: this is not a full corpus comparator replay or a fresh GPU experiment.
The core copy of the full `ExamplePairTests` class still refers to corpus-owned
fixtures; its complete version belongs in VeriTileBench. The retained core
layout check and the three-example CI script were run separately. Numerical
reports and Lean admissions were not regenerated. No version tag or release
was created.
