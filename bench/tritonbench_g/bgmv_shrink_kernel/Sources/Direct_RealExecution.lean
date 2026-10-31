import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Geometry
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Mathematical_RealProgram

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution
open VeriTile.Triton
open scoped BigOperators
namespace Old
abbrev Parameters := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSource.Parameters

end Old
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def parameters (h : Call) : Old.Parameters where
  input := "X"
  weights := "W"
  output := "Y"
  indices := ⟨"Indices"⟩
  N := h.N
  K := h.K
  scaling := h.scaling
  inputBatchStride := h.K
  inputStride := 1
  slotStride := h.N*h.K
  rowStride := h.K
  colStride := 1
  outputBatchStride := h.columns
  outputStride := 1
  BN := h.BN
  BK := 256
  splits := 64

/-- This split's masked summands read immutable entry X/W. The independent
whole-call specification below instead sums directly over Fin K. -/
def contribution (h : Call) (s : BlockState) (b slot n split : Nat) : ℝ :=
  h.scaling * ∑ c ∈ Finset.range h.cycles,∑ e : Fin 256,
    let k := Geometry.rank split c e.val
    if k<h.K then s.readMem "X" (Geometry.inputOffset h b k)*
      s.readMem "W" (Geometry.weightOffset h slot n k) else 0

def writes (h : Call) (p : Geometry.Program h) (slot : Int) (r : RegionName) (off : Nat) : Prop :=
  slot≠-1 ∧ r="Y" ∧ ∃ n : Fin h.N,off=Geometry.outputOffset h p.batch.val n.val

end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.RealExecution
