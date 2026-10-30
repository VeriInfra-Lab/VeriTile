# Benchmark tools

Run from the repository root:

```sh
bash scripts/bench/audit_tritonbench_g.sh --source-only
python3 -B -m scripts.bench.final_inventory --check
python3 -B -m scripts.bench.check_body_corpus --check
python3 -B -m unittest discover -s tests/python -p 'test_*.py'
```

`final_inventory` selects current repaired task entries, enforces the four
benchmark collections, checks imports and binds reviewed source files.
`check_body_corpus` separately preserves Python/Lean correspondence gaps.
`check_imported_kernel_review` guards the 334 imported challenge findings.

`audit_tritonbench_g.sh` without arguments builds and audits the complete
corpus; `--global-only` checks sources and library replay. The full corpus
CI audit runs daily at 04:00 Asia/Singapore or on manual dispatch.
`check_ports.sh` without arguments forwards to the current corpus audit;
named arguments select one task and all its retained Lean helpers.

Shared records are in [validation](../../validation/). Legacy catalog APIs
consume explicitly supplied frozen packages and do not select the live
benchmark implementation. See the [current workflow](../../documents/bench/current_source_acceptance.md).
