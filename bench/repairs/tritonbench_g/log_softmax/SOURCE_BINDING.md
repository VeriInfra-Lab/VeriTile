# Source and proof binding

`Kernels.lean` is the single readable Lean source for the existing
`log_softmax_repaired.py` forward/backward kernels. It preserves all FP32 loads,
identity casts, padding, reductions and stores. Its second source in each pair
retains the pre-existing candidate that exchanges `pid_m * BLOCK_M` with
`BLOCK_M * pid_m`. The selected original Python is unchanged. candidate_kernels.py explicitly labels its authored counterparts of those existing Lean candidates; no performance claim is made.

| Python operation or boundary | Lean source and connection |
| --- | --- |
| All four BM configs and warp-count branches | `Grid.RetainedBlock`, `Grid.warps`; public theorems quantify every retained BM |
| `next_power_of_2(N)` | `Grid.blockColumns`; `block_positive`, `block_covers` |
| Widened program IDs, aranges and runtime sizes | Actual source INT64 casts and three-value tuple assignment; Kernel.signed_index_eq proves strict Nat/Int clamp equality without dropping zero-dimension coverage; `Address.raw_row_exact`, `Machine.native_masks` establish signed64 correspondence on invoked shapes |
| `minimum(raw, size - 1)` before products | Literal `tl.minimum` statements; `KernelGrid.address_active`, `Address.inactive_intermediates` |
| Full row-major offset and 3-part mask | Canonical `offset`/`mask`; `Kernel.active`, `Kernel.address` |
| FP32 masked loads and identity casts | Actual source execution in `Kernel.forward_exec`, `Kernel.backward_exec` |
| Stable forward max/subtract/exp/sum/log/subtract | Source statements; `Kernel.forwardValue`, `KernelGrid.result`; `RealProof.forward_target` gives the independent unshifted formula |
| Backward zero gradient padding, exp, sum, multiply/subtract | Source statements; `Kernel.backwardValue`; `RealProof.backward_target` gives the conventional VJP |
| Default enabled backward FP fusion | `Observed.run` executes the canonical source into `Native.Term`; `NativeGrid.backward_fused` interprets its sub/mul form with the selected `fmaRN` instruction |
| Two-dimensional selected grid | `KernelGrid.programs`, `program_execution`, `chain`, `coverage`, `whole` |
| Empty tensor branch | `KernelGrid.programs` is empty when `M*N*K=0`, so no source execution is required |
| Existing integer-product candidate | `Rewrite.forward_profiled`, `backward_profiled`; public equivalence additionally constructs successful executions |
| Tensor/dtype/dimension/extent guards | `Contract.checkInputs`, `checkBackward`; exact iff theorems |
| Original input/gradient `.contiguous()` calls | `Binding.ContiguousResult` is a local library contract; `prepared_read` derives actual pointer readback from shared `TensorLaws` |
| Observed Tensor pointer and logical element | `Binding.actual_pointer`, `direct_read`; core `VeriTile/Torch/FP32.lean` API citations |
| Caller Tensor values in native-observed source inputs | `NativeGrid.entry`, `InputMemory`, `prepared_input`, `direct_input`; one absolute read memory allows shared or overlapping input storage |
| Actual Tensor values in public Real correctness | `TensorBridge.forward_tensor_correctness`, `TensorBridge.backward_tensor_correctness`; `Binding.PointerInput` at the observed pointer, shared contiguous element laws, and explicit `realState` / `logicalReal` word interpretation derive the existing `InputValues` premises |
| Four-byte physical offsets and exterior frame | Existing `Address`, `Machine`, `Execution` results under explicit local byte read/write and fresh output laws |

The public source equivalence is valid for arbitrary interpretations of the
unchanged scalar/cast/reduction primitives and all FP32 words. It does not
assume agreement between native exp/log and a software reference. The full-grid
FP specialization preserves complete symbolic source graphs, including casts,
and reuses `softmax_flaggems/Proofs/Native.lean` to observe those graphs with
explicit complete reduction trees and local native instructions. Fusion and
operand order are shape-dependent compiler selections shared by the two
variants. The backward sub/mul site may select `fmaRN`, as shown by
`NativeGrid.backward_fused`; no split-versus-fused equality is asserted.
The same native-observed state witnesses constructive execution of each
source. Source-to-compiler schedule selection remains an ordinary lowering
boundary; native contraction is still enabled by the unchanged Python launch.
These model parameters are not deferred numerical experiments.

The Real theorem interprets active input cells as mathematical real values;
masked negative infinity is only a sentinel. It is neither an IEEE decoder nor
a Python finite-value guard. NaN/Inf behavior stays in the all-word FP statement.

Historical full host/autograd results remain optional helpers. The public
kernel results do not require proving allocation, Torch implementation,
autotuner callback internals or wrapper return objects.
