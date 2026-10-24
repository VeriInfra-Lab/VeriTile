# rowwise_quantization_triton

- Source file: `rowwise_quantization_triton.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The accepted floating stage stops before `llrint`: it stores the scaled fp32
row and the binary16 maximum, with independent initial-memory values and a
complete frame. Both parameterized public statements in `FPCorrect.lean` retain
their original hypotheses and scope.

The current software profile uses maxnum instead of the old encoded-rank
maximum. Local Triton 3.8 lowers unindexed `tl.max` through default
`PropagateNan.NONE` / `create_maxnumf`; a single NaN must therefore not replace a
numeric maximum. The old unrestricted rank instance disagreed on an admitted
row containing `1` and NaN. `MaxnumWitness` checks an actual repaired execution:
the numeric output is `127`, the stored half maximum is `1`, the NaN output
remains NaN, and an inactive destination retains its old value.

No finite/non-NaN premise or new padding restriction was added. The maximum
reference now depends on the explicit configuration and ascending maxnum
profile imported from `quantize_copy_kv/Proofs/FPMaxProfile.lean`; this small
module contains no quantizer source execution. This is a new source/profile
revision, not a reinterpretation of an old frozen proof receipt. Native
reduction scheduling, NaN payload/zero choice, scalar arithmetic configuration,
and storage/cast lowering remain binding obligations. The fixed witness only
supports the unchanged parameterized theorem; it does not replace it.
