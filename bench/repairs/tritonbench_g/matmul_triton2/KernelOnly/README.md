# matmul_triton2 kernel-only review

The ledger-selected source remains `revision001/matmul_triton2_repaired.py`, SHA256 `8be06ecb63df7503e779d8dcf10d90de69d378f977b7f2cd9e06aeeb06eaef56`. The frozen revision tree is preserved. This directory supplies new kernel-only endpoints; it changes neither Python admission nor arithmetic.

`IO.lean` gives the shared source-connected `ProgramIO`: two strided matrix inputs, one strided output, all logical coordinates active, no scratch and the complete original grouped launch. `Correct.lean` states `kernel_real` against the independent matrix sum; `Misc/NativeCorrect.lean` preserves the ordinary `FPEquiv.kernel_native` theorem against the ordered padded fixed-k8 instruction target. `FPEquiv.lean` compares the complete selected source with the mask-hoisted candidate described below; optional native results require an explicit Misc import. The mathematical specification and auxiliary native theorem use the shared `ProgramIO.Implements` judgment, including successful execution, actual source access safety, complete outputs and the exact cell frame. The public statements contain no raw states or execution predicates. Original geometry, positive K block size and output/input separation remain explicit; input/input aliasing remains admitted. The eight original configurations, arbitrary initial output contents, K=0 and empty grids are retained.

The Real target is independently defined as

```
C[r,c] = sum(k=0..K-1, A[r,k] * B[k,c]).
```

Its physical-address input reader is arbitrary `Nat → Real`, not a decoder limited to FP32-representable values. The native target uses original raw FP32 input words, an explicit positive-zero initial accumulator, all masked/padded K lanes, ascending K blocks and fixed-width k8 TF32-mode instructions with a live FP32 accumulator. Configurations 0–5 use the captured SM90 warp-group family; configurations 6–7 use synchronous MMA. Two local instruction ports remain model boundaries. No whole-dot, tile, output equality or native-to-Real identity is assumed, and no new primitive numerical experiments are requested.

`Contract.lean` reuses the exact original input policy: same-device CUDA, rank-two strided contiguous FP32 tensors, compatible dimensions below 2^63, resolved negative flags and aligned nonempty input pointers. Empty input pointers remain unconstrained. Contiguous offset views, irrelevant singleton strides, overlapping inputs, K=0 and all FP32 payloads remain admitted. `check_inputs_iff` exposes exact dynamic argument admission.

The ordinary per-kernel output descriptor has shape `[M,N]`, the input device, contiguous zero-offset storage, resolved negative flag, aligned active words and storage disjoint from the two inputs. These are properties of the buffer passed by the wrapper's explicit `torch.empty`, not new input restrictions. The retained `TensorBridge.real` and `TensorBridge.native` applications take actual `Torch.FP32.Tensor` handles and their ordinary API laws. They do not take an allocator function, returned-object law, host invocation, execution result or output equality.

`Tensor.lean` constructs the initial memory from actual pointer addresses and connects native input reads to `Torch.FP32.logicalValue`. Output stores read back through the actual output object's storage address and physical index; the native endpoint states actual `logicalValue` equality. The projection preserves the complete physical exterior and both input storage intervals, including unused backing words. The complete source theorem also preserves every raw tag/cell outside the output element range. No general byte-heap framework is introduced.

`Safety.active_accesses` and `pointer_windows` cover every active masked source load/store, actual storage capacities, alignment and exact uint64 pointer windows over the public kernel contract. Empty grids have identity execution and a complete physical frame; K=0 has a separately audited zero target.

The historical finite executor's `Ready` condition includes `grid_count < 2^31`. `Safety.finite_source` retains this **explicit auxiliary condition** when relating that finite executor to the quotation. It is not part of Python admission, `InputContract`, `KernelContract`, `real` or `native`. No whole-domain finite-executor claim is inferred from this conditional theorem. Shared grid-product/launcher compatibility remains deferred under the existing repository scope; backend resources and arbitrary runtime success are not certified here.

Validation passed 24 original rejection cases and 107 CPU-interpreted cases across all eight configurations. A separate probe captured the actual unchanged wrapper with genuine CPU Torch tensors exposing mocked CUDA admission/context, then used the real JIT binder and `_pack_args`. Twelve offline SM90 compilations passed: all eight configurations for M=N=K=1, two tail cases and two K=0 cases. The three stride-one values are actual constexpr arguments. The source uses `tl.cast` on metadata and strides, so constexpr one never receives a `.to` method call. TTIR confirms a TF32 dot with the live accumulator and no separate floating add; PTX confirms the expected k8 instruction families. This is structural compiler evidence, not GPU numerical conformance.

Only this task's unchanged helper objects were refreshed to the current Algebra layout, followed by successful fresh builds of all ten new modules. No `matmul_triton1` source or object was modified or compiled. `Misc/Audit.lean` checks current load interpretation by `rfl`, standard-base axiom footprints, and non-circular targets/admission/binding/projection. Evidence and receipts are frozen under `.veritile-cache/matmul-triton2-kernel-evidence/`; root owns independent review and ledger integration.

`Proofs/KernelSafety.lean` walks the actual original prelude, every dynamic K-loop iteration and masked output stores. It derives safety from the IO windows using the existing execution invariants; bounds and frames are not substituted for actual source safety. `Proofs/PublicIO.lean` identifies the shared schedule with the prior complete grid, connects logical inputs and reuses `Core.grid_correct`. The old operational statements, including the full-state empty-grid identity, remain ordinary theorems under `RawKernel` in `Misc/RawCorrect.lean` and `Misc/RawNative.lean`.

`Misc/TensorBridge.lean` retains the previous `real`, `native` and Tensor-bound `empty` applications unchanged, including physical exterior and input backing-storage frames. Both the IO and Tensor applications reuse the exact original source execution foundation. Neither public entry imports the optional Tensor bridge. No Python guard, selected source, precision, numerical model or experiment status changes in this format pass.

## Program-pair review, 2026-10-21

`FPEquiv.source_pair_equiv` compares the full selected source with
`matmul_triton2_mask_hoisted.py`. The candidate computes `row_mask` and
`column_mask` before the K loop, reusing them in both loads and the final
store. This removes two invariant vector comparisons per K iteration. No
measured speedup is claimed. The original transparent shared source remains
in `revision001/Kernels.lean`; the complete ordinary candidate quotation is
`maskHoisted` in `KernelOnly/Kernels.lean`. The earlier row-product candidate
remains available but is not the main comparison.

The original FP32 storage, live ordered padded TF32 dot, all uint64 casts,
grouped coordinates, K tails, pointer updates, complete wrapper and all eight
autotune configurations/keys are unchanged. Empty M/N and K=0 remain included.
Each selected configuration is compared with its own corresponding candidate;
the proof does not equate distinct configurations.

`Proofs/MaskExecution.lean` extends the original source invariant with the two
exact cached comparisons and proves successful complete candidate execution
with the unchanged masked-load/dot recurrence and full memory scatter.
`MaskSafety.lean` proves every candidate load and store safe. `Pair.lean`
transports memory equality across the entire actual invocation schedule while
allowing the private mask registers to differ. With precisely the original
geometry, positive blockInner, region separation, input binding and IO bounds,
both full grids are safe and successful, all output cells agree and both exact
exterior raw frames hold. No new finite-input/overflow premise, FP rewrite rule
or numerical experiment is introduced.

The independent Real theorem and optional native/Tensor/raw applications remain
unchanged. `Pair.implements_transfer` also transports any independent original
complete IO formula with safety and its exact frame.
