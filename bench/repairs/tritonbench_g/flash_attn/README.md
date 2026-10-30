# flash_attn

Reference responsibilities (2026-10-23): The shared ordered source recurrence and execution connection are in `KernelOnly/Proofs/Reference.lean`; `KernelOnly/Reference.lean` is an import-only compatibility entry. The selected `flash_attn_scalar_abi_checked` Real and FP entries retain this shared proof dependency, including its original casts and accumulator order.

The checked repair runs directly on ordinary resolved inference FP32 CUDA Q/K/V
of equal positive `[B,H,N,D]` shape, with D in 16/32/64/128, positive independent
strides and independent storage. A sorted stride scan checks nonoverlap and
backing. The caller's boolean causal flag and raw integer or binary64 scale are
retained. Scale must be finite and within FP32 range; zero and negative values
are allowed. No scalar conversion or tensor normalization is added.

The accepted repair kernel AST is unchanged. It retains independent batch/head
bases and V strides, boundary masks and key-tail masks, causal endpoint bounds,
Q scaling then FP16 rounding, probability FP16 rounding, FP32 widening for the
original FP32 K/V dot operands, and terminal FP16 rounding widened to FP32 O.
Both original allocations remain: `empty_like(q)` O and fresh FP32 `[B*H,N]` L.
The wrapper selects q's CUDA device during launch and restores the caller context.

`Contract.lean` models the executed checks and derives shape admission,
nonoverlap, backed accesses, signed32 source arithmetic, natural alignment and
absolute unsigned64 byte bounds. The pointer guard also reserves 512 row strides
beyond the source envelope for masked i64 prefetch pointers in the pinned
four-stage profile. The reserve fits signed64 by the same checked strides.
Bindings contain ordinary Tensor, sort-order, allocation and ABI observations;
they contain no source-bound or result premise.

`Correct.lean` exposes `Checked.whole_real`: the complete independent normalized
base-2 weighted sum for O and log-sum-exp for L, with the full source frame.
Structural safety uses `Checked.structural_safe`, `formed_tensor`,
`formed_prefetch` and `formed_scratch`, independently of numerical correspondence.

`FPEquiv.lean` exposes `Checked.whole_native`. Actual four-byte FP32 loads feed the
precise selected graph, including its fixed K8 instruction chains, four reduction
replicas, FP16 conversions, native FMA sites and scalar exceptional behavior.
The raw scalar payload and its observed native conversion are separate from the
Real projection, preserving signed zero. The theorem reads exact O and L words
through their actual descriptors in a finite 64-bit byte address space, preserves
returned object/storage identity, dtype, shape, strides and device, restores the
caller device context, and proves the full exterior and input-allocation byte
frames. The internal scratch serialization winner remains explicitly quantified.

The source-to-recorded Triton 3.7.1 SM90 graph remains the ordinary compiler/ISA
boundary supported by the eight existing causal/D profiles. No compiler-internals
proof or unconditional native-to-Real numerical law is claimed. Numerical
relations for local instructions remain separate and transport to the actual
returned Tensor through `Checked.returned_relation`.

Validation: 59 actual host-helper/wrapper controls passed, including raw scalar
payloads, allocation retention, pointer checks and device restoration. Eighteen
CPU Triton interpreter controls checked every O/L cell and raw-byte frame across
causal modes, all D, row tails, independent strides, batches/heads, zero and
negative scales. These controls do not establish native GPU numerical conformance.
The original defect witnesses and original Python are retained. Original SHA-256:
`fedc0f296ff2f6c4176ee990c1ea4a29129466998c8ac2fc3f6b1ccc5dfc84a1`.

TODO: the shared Triton launcher signed grid-product compatibility issue remains
deferred by the user. No task-specific product guard is added. The retained
scratch-source extent bound and per-axis bounds remain checked; successful
runtime delivery is the ordinary launch boundary. Native numerical validation
is separate from the structural and byte-return proof.
