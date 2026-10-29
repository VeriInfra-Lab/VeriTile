# Guarded materialization audit fixtures

Read-only snapshots captured 2026-10-13T17:12:17.069226+00:00.
The Python file is the imported TritonBench source at upstream commit 603e28a.
The Lean quotation is the complete layer_norm_fwd macro before audit metadata.
RandomProfile is the reviewed identity-adapter implementation used by the audit.
The production-gate positive fixture selects fp32 for all ten floating storage
arguments while retaining every statement and all ten symbolic flags in the
comparison AST. The definite-assignment check examines all 1,024 flag valuations
without removing any branch from that AST. Two additional unit-test storage
configurations exercise dtype binding and destination casts; they do not establish
coverage of all storage combinations.

These snapshots and source-only harness results are source-comparison evidence.
They do not establish native admission, all-profile coverage, IEEE behavior, or
proof replay. A separate test reads the current actual port through production
source discovery instead of treating the frozen quotation as its acceptance.

| File | SHA-256 |
| --- | --- |
| Kernels.lean.txt | `edacf9764b98ebb7e84940217aed41cecd2adf1ae91a2d14b4f529f517ff0f8a` |
| RandomProfile.lean.txt | `f7704622063dbaed4d3de60e7a9a49d3aaecffcb1949b70c5fbbc3d8f0c02f37` |
| layer_norm_fwd.py.txt | `27dfccf23671d5b9c6f4b749cef39ddf69079793c486a833cbae9c57fe3a4de4` |

Current tests use `RandomProfile-20261013-direct.lean.txt`, SHA-256
`37e7fadbcdfe41dc76c68638ab3ed4215c187752b097f38958d03f1255054467`.
This reviewed frontend adds direct Philox, explicit helper bindings and
unsigned arithmetic; the identity adapter definitions and recognizers are
unchanged. The original adapter file above remains an unmodified historical
snapshot, as do the earlier build receipts.

- `RandomProfile-20261014-signed.lean.txt`: `61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04`. Adds i64-to-i32 source provenance only after actual signed low-word cast lowering. Identity adapters/recognizers are unchanged; older fixtures are retained.
