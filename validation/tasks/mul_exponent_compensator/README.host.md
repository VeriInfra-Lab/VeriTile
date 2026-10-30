# mul_exponent_compensator: direct FP32 repair

The original floor grid omits short inputs, nonmultiple tails, and most coordinates of multidimensional tensors. Its raw-pointer path also misreads noncontiguous/negative views and can wrap int32 offsets. The separate repair retains the accepted `mec_p2` masked kernel: sign-extend PID and lane to int64 before arithmetic, compare against an int64 element count, and mask both load and store. The constexpr factor remains `2.0 ** (127 - 15)` and the operand order remains `x * exponent_compensator`.

The original Python is unchanged (SHA-256 `d1978613929a72a7d01dc152e095143a27abfe01be68071c845e38468b086027`). No flattening, materialization, transfer, or copy is introduced.

## Checked successful domain and host behavior

`src` is a positive-rank, strided, default-contiguous FP32 CUDA Tensor with no unresolved negative bit. The caller block defaults to 1 and must be a Python integer power of two from 1 through `2**20`; ordinary Python bool subclass behavior is retained (`True` means 1, `False` is rejected). The ceiling grid must not exceed `2**31 - 1`. Nonempty input must have a four-byte-aligned actual pointer and be on the current CUDA output device. There is no artificial `numel < 2**31` restriction; widened index bounds follow from block/grid limits and live storage.

The wrapper preserves `torch.empty(src.shape, dtype=torch.float32, device='cuda')`, so output is allocated on the current CUDA device, with input shape. Allocation precedes the original `src.shape[0]` access: rank-zero input still raises `IndexError` after allocation, before block validation. Other tensor/block rejections likewise occur after allocation; only the non-Tensor check precedes it. Empty positive-rank input returns the allocated Tensor without querying input pointers, checking device equality, or launching. Thus empty input on another CUDA device still returns on the current device.

Nonempty input is passed directly to one launch, and the exact allocated output is returned. Ordinary allocation, compilation, resource, or CUDA runtime failures remain runtime failures; this proof is the successful-runtime contract, including for very large caller blocks.

## Proof boundaries

- `Contract.lean` and `Proofs/Contract.lean`: exact acceptance iff checked input contract, including block classification and rank precedence.
- `Kernels.lean`: source-only typed FP32 AST. `Proofs/Source.lean` and `Grid.lean` retain the accepted complete typed execution and coverage proofs.
- `Proofs/Native.lean`: lowers the complete AST, including constant/load/ordered multiply/store wiring. It interprets i32 PID/lane sign-extension before i64 multiply/add, signed masks, and actual u64 byte pointers. All launched lane offsets are below `2**51`; active addresses stay within the live input/output allocations. Generic slot memory, stores, coverage, and arbitrary complete program-order proofs reuse `triton_mul2` helpers.
- `Proofs/Host.lean`: ordinary successful `torch.empty` metadata/freshness/frame facts, shared Torch Tensor logical reader, actual data-pointer identities (including nonzero storage offset), derived input preservation, and allocation-before-rank-failure state.
- `Correct.wrapper_real` and `Correct.typed_real_grid`: independent mathematical `x * 2^112`, exact host output and complete typed-source grid.
- `FPEquiv.wrapper_fp32`: complete native-source result for every FP32 word, exact output identity/shape/current device, input preservation, and every byte outside the output view preserved relative to entry. Empty input uses the same contract. The finite scalar backend is tied to the typed AST by `FPProof.source_scalar`; `constant_exact` proves `0x77800000` denotes `2^112`.

The only deferred numerical claim is the historical `FPProof.LocalMultiplyExact`: one ordered native FP32 `x * 0x77800000` agrees with `FP.mul Config.ieeeValue` for finite input words (nearest even, no input/output FTZ). `wrapper_ieee_if_local` is explicitly conditional on that leaf. Structural all-word correctness imposes no finite-value or NaN-payload law. No host, source, allocation, indexing, pointer, mask, coverage, or frame premise is deferred as a numerical law.

## Validation

All ten task-local Lean modules compile with standard-base axiom audits; no `sorry` or new axiom. The Python source/guard audit and actual JIT AST CPU-storage interpreter pass 19 rejection and 13 admission cases, including reverse program order, tail and multidimensional coverage, singleton strides, nonzero offsets, empty out-of-storage-offset views, unaligned storage rejection, negative-bit rejection, block bool/type/bounds, current-device semantics, rank-zero allocation precedence, exact returned object, and input/exterior bytes. Original omission and int32-wrap counterexamples are retained.

`check_cuda_lowering.py --output-dir /tmp/veritile-mul-exponent-lowering` compiles selected FP32 signatures offline for SM80/SM90 and blocks 1/1024, checking widened int64 indexing/masks, predicated FP32 load/store, and multiplication by `0x77800000` without explicit FTZ. This is compiler evidence, not GPU execution or a proof of the deferred numerical leaf. CUDA hardware is unavailable in this environment; GPU numerical experiments remain pending.
