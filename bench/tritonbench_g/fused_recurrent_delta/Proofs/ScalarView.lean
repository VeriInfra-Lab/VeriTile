import bench.tritonbench_g.fused_recurrent_delta.Kernels
import VeriTile.Triton.DSL.ScalarView

/-! Scalar-specialized mathematical views generated from the actual runtime
source. No kernel body is maintained here. RuntimeBridge.literal_pair checks
the complete execution and access-safety connection to the runtime scalar ABI.
-/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Source
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000

def forward
    (q k v beta o h0 ht : RegionName)
    (s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA : Bool) : ComputeKernel :=
  source_scalar_view
    (Runtime.forward q k v beta o h0 ht s_qk_h s_vo_h B H T K V BK BV
      USE_INITIAL_STATE STORE_FINAL_STATE IS_HEADWISE_BETA) at "scale" := scale

def backward
    (q k v beta dht dh0 do_ dq dk dv dbeta h0 : RegionName)
    (s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ)
    (USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT : Bool) : ComputeKernel :=
  source_scalar_view
    (Runtime.backward q k v beta dht dh0 do_ dq dk dv dbeta h0
      s_qk_h s_vo_h NK B H T K V BK BV
      USE_INITIAL_STATE IS_HEADWISE_BETA USE_DH0 USE_DHT) at "scale" := scale

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Source
