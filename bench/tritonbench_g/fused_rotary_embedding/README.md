# fused_rotary_embedding

- Source file: `fused_rotary_embedding.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Concrete fp32 and payload correctness

`FPCorrect.lean` provides two independent contracts matching the two previously
accepted public sub-kernels. The Python tests explicitly create fp32 Q, K, V,
cosine, sine and caches (`fused_rotary_embedding.py:215–223`). The typed source
uses `DTYPE = .fp32`, retaining fp32 memory and register tags; this source has no
Python register cast or quantization step to erase.

`fused_rotary_q_fp32_correctness` covers the complete selected in-place Q pair
from Python lines 43–63. Every output is independently specified from the
initial-memory Q halves and precomputed cosine/sine inputs. Concrete
`FP.mul`, `FP.sub` and `FP.add` round each source operation separately under
`cfg`; the first result is `sub(mul(q0,c),mul(q1,s))`, the second is
`add(mul(q0,s),mul(q1,c))`. The theorem includes successful typed execution,
both output readbacks and preservation outside both write sets. It retains
the original positive component-stride condition and adds no numerical law,
finite-value restriction, or assumed output. The in-place source loads both
halves before either store.

`fused_rotary_vcache_typed_correctness` covers the complete selected V chain:
KV-group leader guard; sequence length read; block-table lookup; payload load;
and cache store. Its address is derived independently from initial metadata:
`m1 = lengths[pid1]`, `m2 = table[pid1*bts + ((m1-1)/blockSize)*btb]`, then
`m2*cacheBlockStride + (pid0/groups)*cacheHeadStride +
((m1-1)%blockSize)*cacheSlotStride + lane*cacheDimensionStride`. It preserves
arbitrary raw fp32 payload, with no arithmetic assumption. The active-address
injectivity gate is already present inside the original chained IO contract;
nonleaders have no writes. Input/output aliasing is supported at the selected
V-subkernel boundary because the load precedes the store. `FPV.inactive_frame`
proves successful execution with the entire memory unchanged for nonleaders.

Both `*_source_projection` theorems establish exact correspondence with the
entire old selected source after dtype erasure, including statement ordering,
metadata and guards. The output contracts are independent specifications,
not source self-equivalence.

The Q witness has empty initial registers, computes `1 ± 2^-25`, and proves
both results are `0x3f800000` even though the exact rational results differ
from 1. Its second Q half changes from `0x33000000`, and a neighboring cell
is unchanged. The V witness copies signaling-NaN bits `0x7f800001`, negative
zero `0x80000000`, infinity `0x7f800000`, and subnormal `0x00000001`, from
addresses 14–17 to 324–327 through nonzero token/head/block/slot metadata;
address 323 retains its sentinel.

The full fused Q/K/V/cache composition remains outside these accepted
sub-kernel contracts. No quantization scope is added. CUDA/compiler
contraction, FTZ/NaN policy, pointer bounds and machine-address arithmetic
remain a separate native refinement boundary; Q is a concrete software fp32
contract and V is bit-preserving typed execution.

Validation logs: `/tmp/fused-rotary-q-{execution,contract,software}.log`,
`/tmp/fused-rotary-v-{contract,witness}.log`, and
`/tmp/fused-rotary-public.log`. All modules include `#axiomsClean` checks.
