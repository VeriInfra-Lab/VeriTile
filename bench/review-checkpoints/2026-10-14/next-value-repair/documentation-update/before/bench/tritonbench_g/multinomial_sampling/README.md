# multinomial_sampling

- Source file: `multinomial_sampling.py` (upstream `data/TritonBench_G_v1/multinomial_sampling.py`)
- Corpus: TritonBench-G v1
- Size: 136 lines, 1 `@triton.jit` kernel(s)
- Status: source import and complete proof obligations remain pending.
  Public `tl.rand` support is now integrated and its affected dependency closure
  has passed; this does not establish this kernel's source, execution or contracts.
  See [`../tritonbench_coverage.md`](../tritonbench_coverage.md) for the
  ranked unlock levers.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap. It is currently a placeholder: the upstream
Python source has **not** been imported.

The 2026-10-14 frontend work proves signed i32 low-word conversion and Bool-to-integer conversion in all three evaluators. A complete fp32 source-body probe with original i64 seed/offset/index/output storage elaborates in the isolated stage; this is not yet an ordinary paired public source import or a full sampling contract.

The [unchanged-source native checkpoint](../../review-checkpoints/2026-10-14/multinomial-native-original/README.md) records 8/8 offline sm90 compilations (four floating Scores dtypes, i32/i64 runtime scalar ABI) and 40 verified compiler artifacts. No GPU execution or native correctness is claimed. The [initial cast checkpoint](../../review-checkpoints/2026-10-14/multinomial-cast-stage/README.md) records the first ten Lean checks; later context checks also cover nested RNG and address uses.
