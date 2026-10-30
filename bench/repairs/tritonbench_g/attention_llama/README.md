# Checked attention_llama repair

The independent result is natural-exponential attention: each output is the
weighted sum of V divided by the sum of `exp(scale * dot(Q,K))`. Keys satisfy
`key < N`, and additionally `key + start_position <= query` in causal mode.
This source convention requires `start_position == 0` for a complete nonempty
causal grid. It does not implement a prefix-cache offset convention.

The original source is retained in `bench/tritonbench_g/attention_llama`.
The repaired kernel changes only the score mask to the global key coordinate
and places that mask after scaling. Thus tails and zero/negative scales work.
Q/Out remain FP16; K/V remain either FP16 or raw signed E5M2 bytes. The original
`empty_like` allocation, tensor views, natural `exp`, probability cast, output
cast, dot operations, fusion behavior, branches, tile sizes and launch stages
are preserved. No input normalization or copy was introduced.

The caller checks ordinary rank-four resolved CUDA tensors on one device,
matching batch/head/dimension metadata, D16/32/64/128, nonnegative strides,
N>0, ordinary finite scale, boolean causality and nonnegative signed32 position.
Zero batch/head/query cases return the allocated empty output without launch.
Inputs may alias or use noncontiguous/zero-stride views. Ordinary Tensor backing
and the fresh output allocation supply storage capacity and logical-cell
injectivity; no source-access inequality is hidden in those interfaces.

Actual data pointers are checked for natural alignment and absolute unsigned64
bounds. A source-specific signed32 envelope includes scalar arithmetic, actual
strides, padded query/key lanes and the final unused K/V pointer advance.
`Contract.lean` derives finite formed addresses and active allocation bounds.
The per-axis batch/head launch bound is checked. Successful ordinary runtime
launch delivery remains the runtime boundary.

For the byte branch, the caller reads every actual K/V byte with the executed
`((tensor & 0x7c) == 0x7c).any().item()` check and rejects exponent31. The Lean
byte observation interface contains only signed-int8 range and coordinate
memory equalities. `checked_bytes` derives finite E5M2 decoding from successful
checks; it does not assume finite GPU contents in an ordinary binding.
This numerical decoding restriction is absent from structural access safety.
Negative-zero byte encodings are retained. Region-name distinctions distinguish
representation roles; they do not require physically disjoint input allocations.

`Correct.lean` exposes `Checked.attention_fp16`, `Checked.attention_e5m2`,
`Checked.access_safe`, and checked formed/returned-address lemmas. These prove
source execution, the independently specified logical result, physical stores,
and complete complementary opaque-storage frames. The Real interpretation
erases arithmetic rounding while preserving E5M2 decoding. It makes no native
rounded-output-equals-Real claim. Existing original defect witnesses and the
historical source proofs remain separately named.

`FPEquiv.lean` exposes `Checked.attention_native` over actual caller bytes,
input Tensor descriptors and the actual Python-scale/ABI-word marshalling.
The scalar argument retains its integer tag or exact binary64 bits, including
negative zero; the graph consumes the observed FP32 ABI word unchanged. Its
Real projection is separate and never reconstructs the native scalar word.
Its FP16 codec reads two physical input bytes and writes two output bytes.
The proved split/join law, source grid schedule and output injectivity give
byte readback through the actual returned Tensor descriptor. The theorem
preserves allocated-object/storage identity, FP16 dtype, shape, strides, device
and pointer, and proves both the entire exterior byte frame and preservation
of the original input allocation intervals. The finite byte memory is indexed
by unsigned64 addresses. Ordinary input-coordinate/codec/scalar-marshalling and
fresh-allocation contracts contain no attention-result premise.

The root also retains the original
complete typed-source carry-first equivalences. The retained Triton 3.7.1 SM80
graph has BM=BN64, stages2, four warps for D<=64 and eight for D128. It preserves
zero-seeded QK, the normalizer FMA, separately rounded carry product, seeded PV,
RN FP16 casts, explicit lane-specific reduction trees, E5M2 fragment permutations,
and multiply-log2(e)/ex2 lowering. The ordinary pinned compiler/ISA binding is
separate from the deferred fourteen local primitive conformance families and
finite `add_commute` source rewrite. No whole-output numerical premise is used.

Current controls run the actual host helper code against instrumented observations
(313 cases including all 256 byte encodings and signed-zero/underflow ABI controls) and the actual Triton kernel in the
CPU interpreter (30 cases, every output and storage frame). The latter cover
both dtype/causal branches, tails, all head dimensions, independent noncontiguous
strides, K/V aliases, zero/negative scales, raw negative zero, and the original
N65 uniform defect (65/128 versus repaired1). These are not GPU execution or
native numerical conformance tests.

TODO: shared Triton launcher grid-product compatibility is deferred until the
mainline work is complete, per the current task scope. No task-local product
guard was added. Native primitive numerical conformance remains deferred at the
explicit local obligations in `Proofs/NativeLocal.lean`.

The source-to-recorded-SM80-graph association remains the ordinary pinned
Triton 3.7.1 compiler/ISA mapping supported by the retained profile audits.
The new byte/object bridge does not prove compiler internals. Native primitive
transport and the separate Real formula retain their existing interpretations.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `4b63f558e0880e2a16709c82ca130ee9309ea4c75212dd56eae202c8b5027e45`.
The installed Triton JIT chooses the current CUDA device and stream; equal
input devices alone did not select the owning tensor device. The actual source
submission now enters `torch.cuda.device(q.device)` and restores the
ambient caller device. Formal host device/stream observations refer to this
effective context. CPU controls check a cuda:1 launch from an ambient cuda:0
caller and restoration on success and submission exception. Kernel AST and
numerical semantics are unchanged. The prior hash is retained as acceptance
correction evidence; no native GPU/runtime conformance is claimed.
