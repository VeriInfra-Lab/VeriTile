import bench.tritonbench_g.decay_cumsum.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.DecayCumsum.SourcePair
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

def original (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : Program :=
  ⟨Kernels.source kind h p scale, []⟩

def indexCommuted (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : Program :=
  ⟨Kernels.indexCommuted kind h p scale, []⟩

def assumptions : Spec.Assumptions FP.GuardedFragment := []

end
end VeriTile.Bench.Repairs.DecayCumsum.SourcePair
