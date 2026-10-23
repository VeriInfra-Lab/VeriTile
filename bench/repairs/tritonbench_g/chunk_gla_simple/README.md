# chunk_gla_simple direct FP32 repair

`chunk_fwd_o_fn_repaired(h, q, k, v, g, BT, scale)` returns the full output `[B,H,T,V]`. Inputs are ordinary contiguous FP32 CUDA tensors on one device: Q/K `[B,H,T,K]`, V `[B,H,T,V]`, gate `[B,H,T]`, and independently supplied history `[B,H,ceil(T/BT)*K,V]`. Read-only inputs can alias, including overlapping contiguous storage-offset views. Singleton strides need not be canonical. The wrapper passes canonical logical strides directly and makes no input copies.

The original example supplied only one history chunk; the checked wrapper requires the storage actually read by the kernel. `BT` is a power of two from 16 through 1024. Builtin finite scalar values must fit FP32's finite range; the native graph retains the actual runtime FP32 word, including signed zero and conversion underflow. Empty B/H/T/V returns `empty_like(v)` before setup. K=0 and small native dot shapes remain admitted metadata cases; allocation or unsupported native compilation/setup can fail normally. Successful kernel compilation for every admitted shape is not claimed.

The repair replaces undefined boundary block loads with explicit zero-masked pointers and widens pointer/index products and the K-loop bound to signed 64 bits. The source QH/QK accumulation, gate exp and gate difference, causal mask, final SV dot with gated-history carry, scale multiplication, output cast, and four-warp configuration are preserved. Nonempty guards bound storage bytes, CUDA grid axes, and actual aligned FP32 address spans. The launch uses the tensor's owner device and restores the caller's device.

For chunk `n = floor(t/BT)`, the independent Real result is scale times the sum of (1) the Q–history contraction times `exp(g[t])` and (2) the causal within-chunk Q–K–V contraction, weighted by `exp(g[t]-g[s])`. The complete finite formulas, including all feature/time tails, are in the restored mathematical closure and `Host.formula`.

The public Lean endpoints are:

- `Correct.wrapper_correct`: checked host behavior, returned allocation identity, full Real source grid, independent values at every actual returned cell, input preservation, and physical frame.
- `Correct.checked_host`: admission rejection, allocation/output-guard/setup failure, empty return, and owner-device restoration.
- `FPEquiv.wrapper_fp32`: the same checked host behavior and all actual returned native FP32 words.
- `FPEquiv.returned_bytes`: four-byte input decoding at each checked tensor's actual byte pointer, scalar stores to the returned tensor, exact FP32 readback, preservation of input bytes, and every exterior byte.
- `FPEquiv.native_abi`: checked guards imply exact native signed control, masks, and address offsets.
- `FPEquiv.local_numerical_transport`: five fixed scalar arithmetic families—literals, binary/unary operations, casts, and FMA—with a separate generic representation transport.

`ScalarGraph` constructs accumulating-dot nodes before numerical interpretation and expands each into the complete ordered scalar FMA chain. No equality between a separately rounded dot/add and an accumulating dot is assumed. Exp lowers through multiplication by the fixed FP32 log2(e) word and exp2. Padding zero is distinct from the runtime scale tag. Physical input projection uses the source's FP32-profile real view and is proved to expose each original input word. Only output storage freshness is required; historical assumptions requiring disjoint read-only inputs are not used.

`ByteMemory` supplies an explicit little-endian four-byte codec, proves both packing directions, and proves the complete output byte image equal to sequential scalar FP32 stores. `ReturnedBytes.run` consumes the checked source result and stores those words at the returned tensor's actual `byteBase`; it preserves arbitrary bytes outside the output span. Its input theorem derives each byte-base binding from the executable alignment guard and decodes the caller's original bytes directly. The byte bridge adds no numerical interface or input copies. Its concrete `WordPrimitives` adapter fixes raw bit leaves to identity; exact word loading is structural, not a numerical assumption.

The numerical/compiler boundary is the observed Triton 3.7.1 SM80 profile with default FP fusion. The proof establishes the source graph and local numerical transport; it does not certify Triton's compiler or GPU implementation. `check_lowering.py` compiles the actual repaired source offline at BT16/K129/V17, BK64/BV32, four warps, and checks all three live dot carries, scalar FMA, and exp2. It is one profile check, not a universal compiler validation.

Validation: public Lean modules compile with standard base axioms only; `check_contract.py` checks preserved arithmetic/casts/config AST, five zero loads, 68 rejected calls, 9 admitted host controls, and 8 exact CPU interpreter controls over tails, multiple K blocks, and independent per-chunk history. Host controls cover direct aliases, offset/singleton strides, allocation/setup failures, output identity, empty returns, canonical stride arguments, and device restoration. No CUDA GPU was available. The original benchmark Python is unchanged.

## Kernel revision diagnostic

The current canonical Python remains unchanged. Actual wrapper admission and
ordinary native specialization now reproduce small-K compiler failures at the
first dot, including K=0 with nonempty output. Four small-K failures and four
boundary/normal controls are recorded in `SOURCE_DEFECT_KERNEL.md` and the task
delivery packet. This is a source-defect result for the kernel revision, not a
completed proof or numerical-validation deferral. The existing conditional
host/setup-error and model results remain retained within their stated scope.
