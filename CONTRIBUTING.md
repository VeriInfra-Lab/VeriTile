# Contributing

Use English for documentation, comments, identifiers, diagnostics, issues,
and pull requests. Mathematical notation and Unicode regression fixtures
are welcome; do not add translated documentation copies or language selectors.

Use [the roadmap](https://github.com/VeriInfra-Lab/VeriTile/issues/1) for development
directions and [coverage and known limitations](https://github.com/VeriInfra-Lab/VeriTile/issues/2)
for corpus scope and theorem interfaces. Open a separate issue for a concrete
regression or a scoped task that is ready for implementation.

Start with [the code map](documents/CodeOrganization.md). The benchmark
sources, selections and proof records live in [VeriTileBench](https://github.com/VeriInfra-Lab/VeriTileBench).

Before submitting a structural or tooling change, run the relevant tests and:

```sh
lake build VeriTile
bash scripts/check-core-examples.sh
python3 scripts/check_docs.py
```

Run the complete applicable Lean build and proof gates locally when changing
proofs, kernels or their dependencies. See [the verification tools](scripts/README.md)
for the library checks and [the benchmark validation guide](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/current_source_acceptance.md)
for the daily corpus audit.

Reference documents are maintained in `documents/`. Regenerate their website
copies with `python3 site/scripts/sync-docs.py`. Put generated output in
`Logs/` and numerical experiment bundles in their documented ignored results
directory. Historical plans belong in `documents/archive/`; dated receipts
must retain their original source version and validation boundaries.
