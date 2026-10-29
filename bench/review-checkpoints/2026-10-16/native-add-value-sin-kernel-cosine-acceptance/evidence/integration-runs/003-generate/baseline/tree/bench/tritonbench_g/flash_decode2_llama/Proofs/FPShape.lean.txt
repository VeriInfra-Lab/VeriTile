import bench.tritonbench_g.flash_decode2_llama.Proofs.FPSource
import bench.tritonbench_g.flash_decode2_llama.FlashDecode2Llama

/-! Exact algorithm projections for all three originally accepted stages.
The maximum projection uses the actual parsed primitive, and does not assert
floating equivalence with the former comparison/select AST. -/

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
open VeriTile.Triton

theorem normalization_projection (Acc SumExp O : RegionName)
    (a : Row) (z : Scalar) (o : Row) (D : Nat) :
    (normalization Acc SumExp O a z o D).toAlgKernel.body =
      (flash_decode2_llama_normalization_store_kernel Acc SumExp O
        a.batchStride a.headStride a.laneStride z.batchStride z.headStride
        o.batchStride o.headStride o.laneStride D).toAlgKernel.body := by
  rfl

theorem copy_projection (Final O : RegionName) (a o : Row) (D : Nat) :
    (copy Final O a o D).toAlgKernel.body =
      (flash_decode2_llama_final_store_slice Final O a.batchStride a.headStride
        a.laneStride o.batchStride o.headStride o.laneStride D).toAlgKernel.body := by
  rfl

theorem maximum_projection (LogExp Prev Next : RegionName) (block : Nat) (l m : Scalar) :
    (maximum LogExp Prev Next block l m).toAlgKernel.body =
      (flash_decode2_llama_running_max_step_kernel LogExp Prev Next block
        l.batchStride l.headStride m.batchStride m.headStride).toAlgKernel.body := by
  rfl

/-- The original maximum, with no extra conversion event. -/
theorem maximum_primitive_node (LogExp Prev Next : RegionName) (block : Nat) (l m : Scalar) :
    (maximum LogExp Prev Next block l m).surfaceBody[4]? =
      some (.assign .real [] "new_max_logic" (.compute (.alg .fp32
        (.maximumFloat .real .nil (.ref .real [] "tlogic") (.ref .real [] "max_logic"))))) := rfl

#axiomsClean maximum_projection
#axiomsClean maximum_primitive_node

#axiomsClean normalization_projection
#axiomsClean copy_projection
end VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
