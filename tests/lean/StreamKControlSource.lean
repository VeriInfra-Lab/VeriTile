import bench.tritonbench_g.streamk_matmul.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StreamKControlSource
open VeriTile.Triton.ControlSource
open VeriTile.Bench.Repairs.StreamK.TileChecked

private def rejected (result : Except String α) : Bool :=
  match result with | .error _ => Bool.true | .ok _ => Bool.false

def changedMappingBinding : Program := source_program triton {
  pid_m, pid_n = swizzle_tile(tile_id, M, K, N, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
}
/-- Arity alone cannot establish the helper connection: swapping dimensions
must be rejected before it reaches the existing typed mapping interpreter. -/
theorem changed_mapping_binding_rejected :
    rejected (SourceAdapter.decode changedMappingBinding) = Bool.true := rfl

def changedMacBinding : Program := source_program triton {
  mac_loop(A, B, C, M, N, K, partials, pid, stride_am, stride_ak,
    stride_bk, stride_bn, stride_cn, stride_cm, iters_per_tile, start_iter, end_iter,
    BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, GROUP_M)
}
theorem changed_mac_binding_rejected :
    rejected (SourceAdapter.decode changedMacBinding) = Bool.true := rfl

def orderedPrecision : Program := source_program triton {
  acc += tl.dot(a, b)
  acc = tl.cast(acc, tl.float16)
}
/-- Preserve separate dot-add and output conversion in source order. -/
theorem ordered_dot_and_half : SourceAdapter.decode orderedPrecision = .ok [
    .dotAdd "acc" "a" "b", .half "acc"] := rfl

def changedPrecision : Program := source_program triton {
  acc = tl.cast(acc, tl.bfloat16)
}
theorem changed_precision_rejected :
    rejected (SourceAdapter.decode changedPrecision) = Bool.true := rfl

#axiomsClean changed_mapping_binding_rejected
#axiomsClean changed_mac_binding_rejected
#axiomsClean ordered_dot_and_half
#axiomsClean changed_precision_rejected
end VeriTile.Tests.StreamKControlSource
