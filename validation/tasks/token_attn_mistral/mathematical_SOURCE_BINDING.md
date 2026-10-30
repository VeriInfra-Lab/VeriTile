# Candidate source binding

Candidate Python SHA256: `0c5cf97e58ab9539ba589afa4bda50e033222236826001b1ab7019c3d49bbabe`.
Predecessor Python SHA256: `36afb09dac2c46a8049cf250c8c592484147e3d6133ad915e6917d8920c9c553`.

The executable AST check removes the one exact added guard, verifies its position immediately after the empty return, and requires equality of the complete remaining module AST. It separately compares the JIT kernel ASTs. `Kernels.lean` reuses the sole unchanged predecessor `Source.lean`; the delivery packet pins its bytes and complete repository import closure.

| Python operation | Formal binding |
| --- | --- |
| `(batch,heads)` grid and program axes 0/1 | `Grid.fullGrid`, `parameters`, `launch_bounds` |
| Original Tensor pointers and canonical strides | `TensorCall.metadata`, `view`, `tensor_address`, `parameters` |
| Signed metadata and Python integer window | `raw`, `index`, `Header.window`, `Contents`, `signed_index`, `signed_window` |
| Unused `B_Start_Loc` | `Role.unusedStart`, unrestricted word value, complete read frame |
| `max(sequence-window,0)` | `first`, original `windowStart` |
| `cur_head // kv_group_num` | `group`, `group_shape`, `group_bound` |
| P mask `n<attentionLength` | `probability_bound`, `probability_logical`, `NativeTarget.products` |
| Table/V mask `first+n<sequence` | `value_bounds`, `value_logical`, `gather_logical`, `NativeTarget.products` |
| Independent masks and padding | Separate P and V branches, including P-tail `0*NaN` |
| Dynamic 128-token loop | Original `Range.foldRange`, `NativeTarget.step/term` |
| Complete products and sum | Original `Native.reduction`, complete ordered reduction plans |
| Accumulation/output cast | `NativeTarget.step/term`, original `Native.eval` |
| D-lane output stores | `offset_dense`, `complete`, `output_bound` |
| Caller logical output and input preservation | `TensorBridge.kernel_correct`, `TensorBridge.kernel_native`, `logical_read`, `read_exterior` |
| Source and physical exterior frame | `RawFrame`, `observe_frame` |
| Empty return | `check_empty_iff`, `empty_domain_unchanged`, `TensorBridge.empty_call` |
| Live grid-y rejection | `check_launch_iff`, `exact_domain`, `launch_bounds` |

Contents checks and kernel loads interpret the same caller memory, assumed stable during the call. Reader functions introduce no snapshot or copy. Tensor laws describe ordinary storage geometry and impose no output-result equation or allocation separation. Native/compiler conformance boundaries are recorded in README and `numerical-obligations.json`.
