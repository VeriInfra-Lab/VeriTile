# Candidate source binding

Candidate Python SHA256: `35b3502fa9e93d09b5ac05dc8ad59cacd516542a69968f894e4da293869dcd82`.
Predecessor selected Python SHA256: `1182d7c84e373ae328250b0d8c63886d12f8bde5c36279bbcdc131db1de0e020`.
Unchanged source quotation: `../token_attn_reduceV/Source.lean`, SHA256 `be43a6cefdbc7ea812b82b98e3cec8644783ede5b5a33e66e3606aacc00ba28a`.

The executable AST check removes the one exact added guard, verifies its position immediately after the empty return, and requires equality of the complete remaining module AST. It separately checks equality of `_fwd_kernel_token_att2` ASTs. This binds the candidate to the preserved source quotation without introducing a second DSL program.

| Actual Python source | Formal binding |
| --- | --- |
| `grid=(batch,heads)`; PID axes 0,1 | `Grid.fullGrid`, `parameters`, `launch_bounds` |
| Direct P/V/Out/table/metadata pointers | `TensorCall.metadata`, `view`, `tensor_address` |
| `cur_head // kv_group_num` | `group`, `group_shape`, `group_bound` |
| signed `tl.load` of request/start/length | `raw`, `index`, `Contents`, `signed_index` |
| request table gather | `page`, `valueIndex`, `active_bounds` |
| P block address and mask | `probabilityIndex`, `formed32`, `NativeTarget.products` |
| V masked gather and zero padding | `valueIndex`, `active32`, `NativeTarget.products` |
| dynamic `range(0,len,128)` | original `foldRange`, `NativeTarget.step/term` |
| padded product and `tl.sum(...,0)` | original structural `Native.reduction`: complete ordered tree |
| FP32 accumulation and output cast | `NativeTarget.step/term`, original `Native.eval` |
| unmasked D-lane Out stores | `offset_dense`, `complete`, `output_bound` |
| caller logical Tensor readback | `kernel_correct`, `kernel_native`, `logical_read` |
| untouched physical storage and read views | `RawFrame`, `observe_frame`, `read_exterior` |
| empty host return | `check_empty_iff`, `empty_domain_unchanged`, `empty_call` |
| live `head > 65535` rejection | `check_launch_iff`, `exact_domain`, `launch_bounds` |

Python metadata queries and validation are related to the same immutable-for-the-call caller memory used by the kernel. Tensor API storage/pointer/contiguity laws describe the ordinary API boundary and retain original object/allocation identities. No tensor payload is snapshotted or copied; abstract reader functions interpret the existing pointer contents. The source quotation and its interpreter are the formal execution model. Specific compiler and GPU instruction conformance remains explicit as described in README.
