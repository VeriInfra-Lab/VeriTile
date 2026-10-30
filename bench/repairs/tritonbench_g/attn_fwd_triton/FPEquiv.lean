import bench.repairs.tritonbench_g.attn_fwd_triton.Contract
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedNative
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.SourceFPEquiv

/-! Public native endpoint: `Checked.whole_native`. It binds actual caller
FP32 bytes to the selected SM90 graph and returns the allocated BF16 Tensor,
with its exact two-byte words and full exterior/input-allocation byte frame.
`Checked.empty_native_return` covers empty outputs. Historical source-Algebra
and reciprocal APIs are auxiliary components in `SourceFPEquiv`; they are not
native numerical laws. -/
