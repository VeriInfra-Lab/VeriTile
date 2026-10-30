# rope_backward_transform

- Source file: `rope_backward_transform.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Checked FP32 entrypoint

The separate [checked repair](../../repairs/tritonbench_g/rope_backward_transform/README.md)
fixes packed-row and coefficient-column strides while preserving original
preparation, returned views, and storage reuse. Its public whole-call theorems
derive index/pointer bounds from guards and include physical output bytes and
the complete caller backing-byte frame. Original conditional results remain.
