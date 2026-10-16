# Add_value, sin_kernel and cosine_compute integration

Three frozen, independently peer and root reviewed finite source candidates. Target legacy full-source count is 11/183, not completion of Real/FP32/BF16. Integration checks are pending. Numerical conformance, native GPU execution and official comparator success are not established by these source packages. Frozen receipts preserve exact bytes, including earlier failures. The active catalog, supplement, combined review and ledger explicitly use .json.gz; the active numerical queue also uses gzip. Wider existing evidence remains; no new dtype matrix is introduced.

## Observed integration checks

The ledger accepts 11/183 legacy full-source scopes: one verified and ten with registered numerical deferrals; 172 remain pending. These are not Real/FP32/BF16 completion counts. All eight Lean targets exited successfully. The benchmark suite ran 283 tests and the sharding suite ran 38 tests, each exiting 0. The global audit retains its actual exit 1 because scope completeness remains open; comparator gate statuses are recorded individually below. All attempts, including any failures before the selected run, are retained. Existing 961 public records, 239 numerical rules, 2,366 exact bindings and twelve callback tables remain identical, as do 172 unrelated ledger rows.

Global gates: {"baseline": "passed", "catalog_reproduction": "passed", "comparator_preflight": "failed", "deferred_numerics": "passed", "legacy_source": "passed", "library_replay": "failed", "scope_review": "failed", "source_tools": "passed", "structure": "passed", "unchanged_inputs": "passed", "value_metadata": "passed"}.
