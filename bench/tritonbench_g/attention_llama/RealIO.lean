import bench.tritonbench_g.attention_llama.IO
import VeriTile.Triton.Memory.ParameterizedAlgorithmProgramIO
import bench.tritonbench_g.attention_llama.RealMathematical
import bench.tritonbench_g.attention_llama.Proofs.RealSourceTransport

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealIO
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
noncomputable section

@[simp] theorem initial_bound (B : Nat) (c : Config) (scale : ℝ)
    (vals : (io B c scale).Ctx) (s : BlockState) :
    RealScalarInvariant.Bound scale ((io B c scale).initial vals s) := by
  simp [RealScalarInvariant.Bound,io,ParameterizedAlgorithmProgramIO.initial]

@[simp] theorem initial_undef (B : Nat) (c : Config) (scale : ℝ)
    (vals : (io B c scale).Ctx) (s : BlockState) (r : RegionName) (o : Nat) :
    ((io B c scale).initial vals s).undef r o=0 := rfl

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealIO
