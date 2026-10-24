# spinning_lock_reduction

- Source file: `spinning_lock_reduction.py` (upstream `data/TritonBench_G_v1/spinning_lock_reduction.py`)
- Corpus: TritonBench-G v1
- Size: 99 lines, 1 `@triton.jit` kernel(s)
- Status: BLOCKED — not imported. The source has nested `while` loops and a
  return-valued `tl.atomic_cas` in the inner condition. The current DSL supports
  CAS assignments/statements but neither `while` nor atomic condition expansion.

The 2026-10-13 source review checked upstream revision `603e28a`, SHA256
`1542d9675258944fa64fd3688b5ecd1a3116a9c4e277b38294093d133cadf20c`.
Minimal Lean admission checks reject the `while` token and an atomic CAS
condition separately. The wrapper initializes locks to zero; a reducing
program polling `CAS(lock, 1, 1)` cannot make that lock ready by itself. Its
progress depends on other programs publishing partial buffers and exchanging
their lock values. Single-program termination under an initially-ready-lock
assumption would not cover this launch. The nine reduction rounds also retain
the accumulator and never reset the locks, so their interleavings and observed
partial buffers need an explicit specification.

Import the `.py` together with its public Lean entries once ready; the corpus
audit requires matching Python/Lean port counts. The former READY label came
from a `tl.*` token inventory that did not check control-flow support.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.
