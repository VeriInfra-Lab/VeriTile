# Contributing

Use English for documentation, comments, identifiers, diagnostics, issues,
and pull requests. Mathematical notation and Unicode regression fixtures
are welcome; do not add translated documentation copies or language selectors.

Use [the roadmap](https://github.com/VeriInfra-Lab/VeriTile/issues/1) for development
directions and [coverage and known limitations](https://github.com/VeriInfra-Lab/VeriTile/issues/2)
for corpus scope and theorem interfaces. Open a separate issue for a concrete
regression or a scoped task that is ready for implementation.

Start with [the code map](documents/CodeOrganization.md) and
[the benchmark index](bench/README.md). The selected source and public proof
entries are recorded in `bench/tritonbench_g/task_progress.json`; the source
and contract checks are described in [the validation guide](bench/current_source_acceptance.md).

Before submitting a structural or tooling change, run the relevant tests and:

```sh
bash bench/audit_tritonbench_g.sh --source-only
python3 scripts/check_docs.py
```

Run the complete applicable Lean build and proof gates locally when changing
proofs, kernels or their dependencies. Daily CI performs the exhaustive corpus
audit; a source-only check does not establish proof acceptance. See
[the verification tools](scripts/README.md) for comparator setup and scopes.

Reference documents are maintained in `documents/`. Regenerate their website
copies with `python3 site/scripts/sync-docs.py`. Put generated output in
`Logs/` and numerical experiment bundles in their documented ignored results
directory. Historical plans belong in `documents/archive/`; dated receipts
must retain their original source version and validation boundaries.
