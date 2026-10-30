# LoRA expand GEMV

The repair checks and launches the original kernel directly on the caller's
buffers. It selects FP32 X/Y, FP16 W and signed int32/int64 indices. Both add
modes, sentinel -1, K tails, singleton 4D weights, extra X/Y rows and Y columns,
aligned storage-offset views and read/read aliases are supported. Y's backing
is separate from X, W and indices. The same supplied output object/backing is
retained and the function returns `None`.

The wrapper validates ranks, dtype, CUDA device, contiguous/nonlazy storage,
scalar type, dimensions, adapter contents, pointer alignment, CUDA y-grid
limit and all formed signed-int32 offsets. It retains the kernel's 256-row
tile; K is 1..4096 so the tile stays within 1,048,576 elements. The installed
six-shift `next_power_of_2` helper is verified over that whole interval in Lean.
The native integer restrictions come from the unchanged source's int32
arithmetic, including masked rows. K=0 rejects before its division by zero.

No tensor snapshot, metadata tensor copy, output allocation or copyback
occurs. Original indices are inspected with `tolist` and passed directly to
the launch. The optional singleton squeeze is an alias. Canonical contiguous
strides are passed under the actual CUDA device context. Zero batches return
before metadata inspection; zero columns return after validating indices.

`Correct.checked_real` derives complete symbolic-source execution and the
independent finite K-term contraction from `Contract.Admitted`.
`FPEquiv.checked_fp` and `checked_caller_bytes` preserve the explicit FP16
products/reduction tree, conversions, compiler operand routing and supported
half mul/add contractions, with exact four-byte caller readback and complement
frames. The native implementation reads final source memory and is proved
equal to an independent reference graph. Whole original read backings remain
unchanged. Failure frames allow any issued subset/order with arbitrary store
payloads; numerical success is not asserted on failure.

Only `Kernels`, `Contract`, `Correct`, and `FPEquiv` are root public Lean entries.
Helpers and historical proof statements are under `Proofs`; the Real and FP
public import closures are independent. Historical source, grid, host, physical
and numerical proofs are retained. The new public result derives `Valid` from
concrete checks and has no whole-host preparation or result-equality premise.

Run `python bench/repairs/tritonbench_g/lora_expand_gemv/check_contract.py`.
The checker verifies production/source hashes, unchanged kernel AST, installed
Triton 3.7.1 reduction/helper selection, 25 complete calls, 38 rejections,
forward/reverse grid orders, direct identities, untouched backing bytes, context restoration and partial
failure. CPU AST execution is not GPU primitive conformance.
