# triton_attention checked repair

The executable remains `Whole001/triton_attention_repaired.py`. Its canonical `attention` entry selects the source's default FP32 domain. `attention_compatibility` separately retains the historical FP16 route; no FP32-only native theorem is advertised for that route. The three accepted repair kernel ASTs and the original benchmark source are unchanged.

The public Lean roots are `Kernels.lean`, `Correct.lean`, and `FPEquiv.lean`. Their checked results live in `VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked`; all historical `Whole001` and original-source APIs remain available.

## Executed contract and original behavior

The wrapper checks exact ordinary resolved strided CUDA Tensor inputs, matching FP32 dtype/device/shape, canonical Q/K/V strides, positive batch/head/sequence dimensions, a sequence length divisible by 128, and head dimension 16, 32, 64, or 128. It retains the source signed32 element bound and CUDA grid-y bound. Scale is an ordinary Python int/float that remains finite and positive under FP32 conversion. The original scalar object is passed unchanged and saved unchanged; the ABI observation records its actual source operand conversion, independently of the real-valued projection.

Before launch, each actual source tensor is checked for storage extent, natural alignment, and absolute unsigned64 representability through the formed-address envelope, including a 65536-element final-advance/pinned-prefetch reserve. Structural lemmas derive storage and formed-byte bounds from these Boolean checks, independently of any floating-point assumptions. Original source row/block/reset bounds remain available. No GPU-content assumptions or scans are required.

Forward retains its O, L, and M allocations and saves the same Q/K/V/O/L/M objects. Backward retains the original `do.contiguous()` operation before `zeros_like` DQ, DK/DV/NewDO/Delta allocation, preprocessing, and backward. A contiguous incoming Tensor may be returned unchanged, including arbitrary singleton strides; a noncontiguous incoming gradient is copied without changing its logical bits. The proof handles both outcomes. No added input copy, value normalization, scratch removal, or numerical guard changes this order.

Both launch blocks execute inside `torch.cuda.device(q.device)`, restoring the prior current device afterward. Q/K/V may alias each other. Ordinary fresh output allocation separation is recorded as allocation observations; read-only input aliasing is not banned.

## What the checked results establish

- `structural_safety`: signed source offsets, backed active accesses, natural alignment, and finite absolute formed addresses, derived without numerical premises.
- `whole_real`: every batch/head/row/channel of erased-real O/DQ/DK/DV equals the independent causal attention and gradient formulas. The same backward results satisfy the derivative/VJP theorem for simultaneous Q/K/V perturbations. The main specification takes arbitrary real values indexed by actual Tensor pointers/strides, and an arbitrary real scale; it has no finite-word representability restriction. `whole_real_decoded` retains the separate decoded-word instance. Neither result identifies native arithmetic with real arithmetic.
- `forward_native`: an independent forward endpoint with only the six existing forward objects, no future cotangent or backward allocation. It returns the actual O object and actual saved object list, raw scalar, owning-device launch/restoration, exact native O/L/M reads, input allocation preservation, and a complete physical byte frame.
- `backward_native`: a later backward endpoint using actual saved-byte views and the actual incoming gradient. It proves DQ/DK/DV/None identity, exact native gradient reads, all scratch/write bytes, current-device restoration, and complete frames over the original gradient and saved allocations. Copy identity is proved for all bit patterns, including signed zeros and NaN payloads.
- `whole_native`: composes those endpoints, deriving the saved snapshot by reading the actual preceding forward stores. All four returned tensors are read from final physical memory and match input-derived native references. The conclusion also frames all global bytes outside the exact forward/copy/backward write extents, preserves original Q/K/V storage, and preserves the incoming gradient relative to the backward entry.

Global memory is finite `Fin (2^64) → BitVec 8`. Descriptor reads use actual data pointers, storage offsets, and Tensor strides; FP32 stores encode/decode four little-endian bytes with both round trips proved. Neither numerical conformance nor allocation bindings contain a final-output oracle. Bindings observe ordinary Tensor metadata, actual scalar marshalling, local instruction implementations, reduction owners, allocation separation, and successful runtime delivery. Ordinary Tensor/storage/allocation/copy/compiler/runtime interpretation is trusted.

The native numerical boundary is the historical eleven local instruction families, including fixed K8 WGMMA with its live-seed operand. The graph keeps FP16 narrowing/widening, native reduction replicas, current DQ reloads, FMA placement, and original source precision. `LocalConformance actual reference` relates only those local operations. It is separate from real mathematics and from every structural/frame proof. CPU interpreter comparisons do not establish native GPU numerical conformance.

## Validation and retained limits

`test_checked_contract.py` executes the actual host helpers/entry methods with allocation/copy/launch spies. Its 55 controls cover admission/rejection, raw scalar identity, original allocations and gradient copy, singleton-stride copy identity, checked FP32 versus separately named FP16 compatibility, output/context/gradient identities, device selection/restoration, and exact original/kernel AST pins.

`TRITON_INTERPRET=1 python bench/repairs/tritonbench_g/triton_attention/test_source_controls.py` runs all three kernels in 11 source controls: all supported head dimensions, 128/256/384 rows, multiple batches/heads, Q/V aliasing, noncontiguous gradients, and singleton-stride copy identity. It checks every O/DQ/DK/DV/L/M/NewDO/Delta cell against independent dense attention/autograd or stage arithmetic and checks complete raw-byte frames. These are source diagnostics on CPU, not a GPU conformance certificate.

The original source SHA256 is `3eb3038c22a9816c1cef15262c8f0b1adcf4f2f960b1235859211f8ed81a1a38`. The three accepted repair kernel AST digest is `7919e8651fef50ea9a4827d57a87e768ae4d2cdb0bd5f12262841754c195e93d`.

**TODO (shared launcher):** The common Triton launcher's signed grid-product issue is deferred by the user. This task retains source/per-axis checks and ordinary successful runtime delivery; it does not add a task-specific grid-product guard. Historical pinned compiler/numerical evidence and defect witnesses remain unchanged under the original and `Whole001` modules.
