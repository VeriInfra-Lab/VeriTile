---
title: Reference
description: Build instructions, scripts, and external resources for working with VeriTile.
---

Operational reference. Most of these point back to the repository itself
since they describe the build and tooling rather than the design.

## Build & toolchain

### Setup

Install Git and follow the [official elan installation instructions](https://github.com/leanprover/elan#installation).
Open a fresh terminal so that `lake` is available on your PATH. Elan selects
the Lean version pinned by this checkout's `lean-toolchain` file.

```bash
git clone https://github.com/VeriInfra-Lab/VeriTile.git
cd VeriTile
lake exe cache get
lake build bench.examples.VectorAdd.RealCorrect bench.examples.VectorAdd.FPEquiv
```

The first build needs network access to fetch the toolchain and dependencies.
Budget several gigabytes for Lean/Mathlib caches and build output. The September
2026 audit used about 11 GB including website dependencies and recorded an
approximately eight-minute library build after the Mathlib cache download;
these are one machine's measurements, not a timing guarantee.
The final command should exit with code 0 and report that the example's axiom
and statement-surface checks pass. Existing linter warnings may still appear.

### Build targets

- **Lean toolchain**: `v4.29.0`. Pinned in
  [`lean-toolchain`](https://github.com/VeriInfra-Lab/VeriTile/blob/main/lean-toolchain).
- **Routine build**: `lake build` from the repo root (the default `VeriTile` target).
- **Full build**: `lake build VeriTile VeriTileFull`, including the GeLU analysis and library trust report.
- **Worked example**: `lake build bench.examples.VectorAdd.RealCorrect bench.examples.VectorAdd.FPEquiv`; this also builds its helper imports. `lake env lean` can then re-elaborate an entry to print its diagnostics.
- **Core CI examples**: `bash scripts/check-core-examples.sh` checks source correspondence and Real/FP builds for VectorAdd, RowWiseSum and LogExp.
- **Manifest + sorry check**: `scripts/check-artifact.sh`.
- **Complete benchmark audit**: in a VeriTileBench checkout, `bash scripts/bench/audit_tritonbench_g.sh`. Its full scope is separate from core CI.
- **Trust checks**: see the [trust audit guide](/VeriTile/proofs/trust-audit/).

## Sub-project READMEs

These live in the repo and are the authoritative source for their respective
areas:

- [Top-level README](https://github.com/VeriInfra-Lab/VeriTile/blob/main/README.md)
  — quick-start + theorem-surface chooser.
- [`bench/tritonbench_g/` README](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/documents/bench/tritonbench_g.md)
  — bench layout, port checklist, kernel inventory.
- [`scripts/` README](https://github.com/VeriInfra-Lab/VeriTile/blob/main/scripts/README.md)
  — what each script does and how CI uses them.
- [`verso/` README](https://github.com/VeriInfra-Lab/VeriTile/blob/main/verso/README.md)
  — the architecture-overview slide deck (separate sub-project).
- [`documents/` index](https://github.com/VeriInfra-Lab/VeriTile/blob/main/documents/README.md)
  — the original markdown design notes (this site re-renders them under
  [Architecture](/VeriTile/architecture/) and [Proofs](/VeriTile/proofs/)).
