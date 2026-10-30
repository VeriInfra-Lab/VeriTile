# rope_transform

- Source file: `rope_transform.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Checked FP32 entrypoint

The separate [checked repair](../../repairs/tritonbench_g/rope_transform/README.md)
fixes the singleton-sequence row stride while preserving the original kernel,
preparation, returned objects, and storage reuse. Its public statements derive
finite ABI bounds from executable checks and include physical output bytes and
the caller frame. Original conditional proof surfaces remain unchanged.
