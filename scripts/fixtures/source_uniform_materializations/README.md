# Complete uniform source fixture

Captured on 2026-10-13 UTC from the actual `uniform_sampling` port and shared
frontend. The Python file comes from pinned TritonBench revision `603e28a`.
The Lean fixture is the complete original/candidate macro and storage selector,
including all original kernel statements. Its initial descriptive header is
replaced by the test's exact selected-source metadata, without editing the body.

| File | SHA-256 |
| --- | --- |
| `uniform_sampling.py.txt` | `d2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1` |
| `Kernels.lean.txt` | `67ba816fa56a773feb128b8b424529a7ba2b8aae1f4086745c0876959832e43c` |
| `RandomProfile.lean.txt` | `37e7fadbcdfe41dc76c68638ab3ed4215c187752b097f38958d03f1255054467` |
| `SourceDomain.lean.txt` | `af632e51119b3d778eef8e847e800ffd01f9adb199f571b874c1bde7476cd2e3` |

The profile must preserve the complete source module's helper `try`/`except`,
both fallback branches, heuristics, decorators, host wrapper and all 20 kernel
statements. Only three identity assignments are inserted for comparison.
The two initial signed parameter casts consume those new bindings; later reads
still use the original rebound parameter registers. `to=upper` is an explicit
injective parameter correspondence required by the Lean identifier spelling.
All four cache hints remain literal fields in the full AST comparison.

These fixtures support selected-source comparison and adversarial regression.
They do not establish proof replay, native helper equivalence, all storage/ABI
profiles, GPU execution or numeric comparator results. Production source tests
run the unchanged extraction and 23 source gates in a source-only legacy-layout
harness; they never create theorem receipts.
The harness copies the actual frozen `Proofs/SourceDomain.lean` to preserve the
macro's complete local import graph. Copying that source is not proof replay.

- `RandomProfile-20261014-signed.lean.txt`: `61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04`. Adds i64-to-i32 source provenance only after actual signed low-word cast lowering. Identity adapters/recognizers are unchanged; older fixtures are retained. The corresponding stage proof and full-source admission receipt is `bench/review-checkpoints/2026-10-14/multinomial-cast-stage`.
