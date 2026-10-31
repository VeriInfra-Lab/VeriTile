import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealLoop
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure

/-! Decomposition of the entire precision-erased source. The signed metadata
load, sentinel guard, split-K loop and computed store/atomic choice remain
part of the program; internal ASTs are connected to that source below. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSource
open VeriTile.Triton
set_option maxHeartbeats 2200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Parameters where
  input : RegionName
  weights : RegionName
  output : RegionName
  indices : Region .int
  N : Nat
  K : Nat
  scaling : ℝ
  inputBatchStride : Nat
  inputStride : Nat
  slotStride : Nat
  rowStride : Nat
  colStride : Nat
  outputBatchStride : Nat
  outputStride : Nat
  BN : Nat
  BK : Nat
  splits : Nat

namespace Parameters

def guard : Op .bool [] :=
  .ne .int .nil (.ref .int [] "lora_index") (.constInt (-1))

def weightPointer (p : Parameters) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase p.weights)
    (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat p.slotStride))
      (.ref .int [] "lora_index")))

end Parameters
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSource
