# chunk_gated_attention FP32 repair

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `KernelOnly/Misc/TensorBridge.lean`, `KernelOnly/Misc/NativeCorrect.lean`, `Misc/InterfaceChecks.lean`.

The selected kernel-only public entry points are documented in
[KernelOnly/README.md](KernelOnly/README.md). They expose the independent Real finite IO specifications in
`KernelOnly/Correct.lean` and actual source-pair equivalences in
`KernelOnly/FPEquiv.lean`, with
the complete Tensor applications in `KernelOnly/Misc/TensorBridge.lean` and reader
audits in task-root `Misc/InterfaceChecks.lean`. The historical whole-host results
described below remain available unchanged.

The two original APIs remain separate. `fwd_pre_repaired(g,B,H,T,S,BT)` returns chunk-local cumulative gates `[B,H,T,S]`. `fwd_inner_repaired(q,k,v,g,B,H,T,K,V,BT,BK,BV,gatek=False,h0=None,ht=None)` returns pre-update history `[B,H,ceil(T/BT)*K,V]` and, when supplied, copies the final state into the original `ht` object. The inner API accepts an independently supplied cumulative gate tensor.

Inputs are ordinary contiguous FP32 CUDA tensors on one device. Q/K have shape `[B,H,T,K]`, V has `[B,H,T,V]`, G has `[B,H,T,K]` or `[B,H,T,V]` according to `gatek`, and optional seed/final tensors have `[B,H,K,V]`. Read-only tensors may alias, including overlapping contiguous storage-offset views. Harmless singleton strides are admitted; the wrapper passes canonical logical strides. Dimensions are nonnegative builtin integers, `gatek` is a builtin bool, and block sizes are powers of two at least 16. The original dot tiles must fit Triton's `2^20` element limit. Executable guards check signed 64-bit byte extents and naturally aligned, nonwrapping actual FP32 data-pointer spans.

The wrapper reuses inputs whose shapes already fill their tiles. Other operands receive only the padding needed by the complete-tile source. Zero padding handles time/feature tails; padded cumulative gate rows repeat the last valid cumulative row. Optional final output always uses a private kernel buffer, then copies back after every launch has finished reading inputs. Consequently `ht` may alias K, V, G, Q or the seed. The empty inner path retains the actual FP32 zero-plus-seed operation, including its signed-zero behavior. Device contexts restore the caller's device on success or failure.

Both kernel bodies retain their floating statements and casts. The repair widens index products, loop control, and flattened program IDs to signed 64 bits, rebases block pointers to zero local offsets, and splits large launch grids into legal batches. All nine original prefix autotune configurations remain. Each selected BS16/32/64 configuration uses the original BS16 outer batch bases; exhausted configurations execute a guarded dummy with no memory effects. No FP16/BF16 or arbitrary-strided domain is added.

The public proofs are:

- `Correct.inner_returned` and `Correct.prefix_returned`: the checked return object, complete Real source grids, independent formulas at actual returned scalar addresses, optional original-target final readback, and exterior frame. The inner formula is a valid-time gated contraction with subsequent gate products; history includes every pre-update chunk state.
- `FPEquiv.inner_returned` and `FPEquiv.prefix_returned`: the checked host, every returned four-byte FP32 word, actual original-byte input projections, native pointer/grid guards, delayed final copy-back, caller-byte preservation, output shape/device, and device restoration.
- `FPEquiv.exterior_inner` and `FPEquiv.exterior_prefix`: every byte outside the explicit padding, kernel-output, returned-output and optional final-copy spans is preserved. Caller input spans have the stronger frame except where they overlap the requested final target.
- `FPEquiv.shared_inner_return` and `FPEquiv.shared_prefix_return`: when slice/reshape/contiguous returns a shared contiguous view, its coordinate map is the identity and the modeled copy changes no byte.
- `FPEquiv.native_range`: signed 64-bit recurrent loop start, stop, comparison, and final excluded increment follow from checked extents.

`NativeValue`, `NativePrefix` and `NativeGrid` construct complete independent source graphs. The recurrent graph retains zero-plus-seed, all gate operations and casts, history stores, final stores, and each live dot accumulator. `ScalarNative` expands every IEEE dot into ordered scalar FMAs; exp lowers through the fixed FP32 log2(e) word and exp2. There is no assumed equality between a separately rounded dot/add and a fused accumulation. The concrete numerical adapter has five fixed scalar arithmetic families: literal conversion, binary operations, unary operations, casts, and FMA. Raw bit leaves are structurally the identity. There is no whole-dot, reduction, tensor-output, or execution numerical port.

Ordinary Tensor allocation, zero fill, coordinate copy, slicing and contiguous-return laws are explicit local storage contracts. They contain no kernel result or successful source-execution premise. Direct input identity is part of the storage binding; historical requirements that all read-only inputs be private or mutually disjoint are bypassed. Native signed addresses derive from executable guards and are tied to those same prepared buffers. The four-byte codec and sequential scalar-store laws are consumed from the local `chunk_gla_simple` bridge.

The compiler/numerical boundary is the observed Triton 3.7.1 SM80 profile with default fusion. Offline checks compile the actual repaired source for both gate branches at BT/BK/BV16, two chunks, seed/final enabled and two warps, and prefix BT/BS16. They confirm the live gated dot carry, zero-plus-seed, scalar FMA and exp2. These are narrow lowering checks, not a compiler correctness proof or a guarantee that every admitted shape compiles on every device; allocation/setup failures remain ordinary host outcomes.

Validation passed: public Lean axiom audits use only the standard base; 19 original floating assignment/store ASTs and all nine autotune configurations are preserved; 45 metadata rejection cases and 24 exact CPU controls cover both APIs, all BS choices, both gate directions, tails, multiple chunks, optional seed/final, aliased seed/final and V/final, direct aligned operands, empty signed zeros, and device restoration. CPU controls execute the actual wrapper allocations/views/copies and original arithmetic, adapting only the CUDA admission/context shell. No CUDA GPU was available. The unchanged original Python SHA256 is `8f67a536f4c7850f6025debc90d692e3d8e007159e92df38ea144c20eac77b1f`.
