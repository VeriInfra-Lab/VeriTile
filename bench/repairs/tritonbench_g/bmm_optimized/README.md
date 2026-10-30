# Checked bmm_optimized repair

The separate candidate zero-fills masked A/B loads while retaining the fully
divisible unmasked paths, all 12 original configurations, `allow_tf32=False`,
default fusion, FP32 accumulation and FP32 output. The original `UndefinedTail`
and unresolved-negative-view defects remain recorded against the unchanged
original Python. Its intentional `A.contiguous()` and `B.contiguous()` calls,
logging and CUDA device-context restoration are preserved.

The checked domain is same-device rank-three strided FP32 CUDA A[B,M,K] and
B[B,K,N], matching batch/K, resolved raw views, nonnegative strides and naturally
four-byte-aligned nonempty data pointers. Valid offsets, broadcasts, transposes
and read-only A/B storage aliases are admitted through the original contiguous
normalization. No new resolution/copy operation is introduced. Empty batch/M/N
returns a fresh empty result; K=0 runs the zero-accumulator kernel path.

The exact accepted signed32 envelope is checked before normalization/allocation:
B, M+128, N+128, K+64; 2*ceil(M/32)*ceil(N/32)+2; (B*M+128)*K+64;
(B*K+64)*N+128; and (B*M+128)*N+128 are each below 2^31. These source-specific
bounds cover all swizzle and padded lane expressions, including the final
pointer advance. The packed kernel never uses caller strides. Ordinary valid
view spans supply active caller-read coverage; `Checked.covered` proves it
without requiring positive strides or distinct read-only input allocations.
Shared grid-product launcher compatibility remains deferred.

`Checked.checked_real`, `checked_native` and `checked_floating` expose complete
returned-object metadata, every output cell/byte, caller frame and the derived
source index domain. They reuse the accepted whole-grid, ownership, arithmetic,
scatter, host and physical-byte proofs. `prepare_raw` proves that rejecting lazy
negative views makes the existing contiguous preparation independent of any
negation operation. `chain_raw_equiv` reduces floating correspondence to one
scalar `fma.rn.f32` numerical family. The target is the ascending-K continuous
live-carry FMA sequence through 32*ceil(K/32), retaining padded zero operations.
It does not replace that observed graph with a separately rounded dot-plus-add.
The Real interpretation independently proves the logical finite-K matrix sum.

All 17 restored closure modules and the checked entry additions compile with
axiom audits. `python bench/repairs/tritonbench_g/bmm_optimized/check_contract.py`
passes 43 actual-source Triton CPU-interpreter executions across the preserved
configuration/stage pairs, K tails, K=0, empty outputs, offsets, transposes,
broadcasts and input aliases, plus five rejected-input classes. It checks the
original contiguous calls and caller frames. These controls do not validate
GPU arithmetic. Compiler selection/fragment routing, ordinary Tensor/storage,
allocation, device context and successful runtime remain realization boundaries;
local scalar numerical conformance remains deferred. No acceptance record is
changed by this task owner.
