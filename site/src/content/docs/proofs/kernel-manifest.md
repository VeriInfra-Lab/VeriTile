---
title: "Kernel Manifest"
---

`scripts/kernel-manifest.tsv` is the canonical per-kernel registry for the
artifact. It replaces the old flat theorem and examples lists.

Each non-comment row has this tab-separated schema:

```text
id	file	theorem	kind	status	source	source_ref	config	label	notes
```

## Fields

- `id`: stable machine-readable identifier. Must be unique.
- `file`: Lean source file containing the theorem.
- `theorem`: public theorem symbol guarded by artifact checks.
- `kind`: theorem category.
- `status`: verification status.
- `source`: kernel/source origin, such as `internal`, `tutorial`,
  `paper:<name>`, or `tritonbench:<path>`.
- `source_ref`: source commit, URL, paper anchor, or `-` when not applicable.
- `config`: important static configuration, such as `BLOCK_N=128` or
  `S=T=D=1,numIters=0`.
- `label`: short human-readable name.
- `notes`: simplifications, dropped features, or other caveats.

Allowed `kind` values:

```text
correct refine math launch trace safety frame
```

Allowed `status` values:

```text
proven projected test-gap blocked smoke
```

## Adding an Entry

For a shared-library theorem selected by the artifact gate:

1. State the public result with its actual model, conditions and scope. Library
   examples may use `ComputeCorrect.*` / `ComputeRefine.*`; worked Real and FP
   headlines follow [CorrectnessSurfaces.md](/VeriTile/proofs/correctness-surfaces/).
2. Add a `proven` row under `VeriTile/` to `scripts/kernel-manifest.tsv`.
3. Pin the source/configuration and record limitations in `notes`.
4. Run `scripts/check-artifact.sh` with the comparator tools installed.

The library comparator selects only proven `VeriTile/` rows and their recursive
theorem dependencies. A metadata row for a standalone example does not make it
part of that replay. Audit an example with `check_comparator.py --file ... --trust`;
VeriTileBench owns benchmark selections, per-module receipts and corpus audits.
Metadata validation, compilation and independent proof replay remain distinct.

The artifact checker validates that every manifest file exists, every theorem
symbol exists, ids are unique, and `kind` / `status` use the allowed
vocabulary.

## Relationship to TritonBench-G

In VeriTileBench, for TritonBench-G ports, `source` should identify the benchmark entry
and `source_ref` should pin the upstream commit or URL. `config` records the
chosen `BLOCK_*` / dtype / static meta-parameters. `notes` records dropped or
simplified features, such as dropout, unsupported async paths, or deferred IEEE
compute semantics.
