# Mul2, ReLU, custom vector and add_example integration

Four frozen, independently peer and root reviewed finite source candidates. Integration checks are pending. Numerical conformance, native GPU execution and official comparator success are not established by these source packages. Frozen receipts preserve exact bytes, including earlier failures. The active catalog, supplement, combined review and ledger explicitly use .json.gz; the numerical queue remains plain JSON.

The user subsequently selected Real / FP32 / BF16 as three primary versions,
with only source-required additional types. This batch retains already completed
wider source proofs under their existing schema2 reviews. Its source-scope
counts do not assert completed per-version review under the new direction.
The direction is recorded in evidence/user-direction-three-versions.json; a
separate version inventory and acceptance mapping will follow.

## Approved primary-version scope

The user confirmed Real / FP32 / BF16 as the three primary versions, retaining only additional types required by the source. The [primary-version policy](primary-version-policy.md) records that decision. Shared proofs are reused with a reviewed version instantiation; no exhaustive runtime dtype matrix is required for new work. This scope change does not promote the in-flight legacy acceptance rows or establish new per-version completion counts.

The [183-task evidence inventory and eight-task mapping](evidence/three-version-inventory/README.md) preserve the committed baseline and distinguish original rejection from a functional FP32/BF16 profile. They perform no new acceptance transitions.

## Observed integration checks

The ledger accepts 8/183 source scopes: one verified and seven with registered numerical deferrals; 175 remain pending. All eight Lean targets exited successfully. The benchmark suite ran 269 tests and the sharding suite ran 38 tests, each exiting 0. The global audit retains its actual exit 1 because scope completeness remains open; comparator gate statuses are recorded individually below. All attempts, including any failures before the selected run, are retained. Existing 957 public records, 153 numerical rules, 1,642 exact bindings and eight callback tables remain identical, as do 175 unrelated ledger rows.

Global gates: {"baseline": "passed", "catalog_reproduction": "passed", "comparator_preflight": "failed", "deferred_numerics": "passed", "legacy_source": "passed", "library_replay": "failed", "scope_review": "failed", "source_tools": "passed", "structure": "passed", "unchanged_inputs": "passed", "value_metadata": "passed"}.
