import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNative

/-! Shared native word observer for the public stage results and optional forward observation. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Triton FP.Structural SourceNative SourceNative.Native
noncomputable section

def observeWords (I : Instructions) (S : OutputSchedule) (r : RegionName) (o : Nat) (term : Term) : Term :=
  .bits (FastLayernorm.KernelNative.eval I (S r o) (FastLayernorm.KernelNative.normalize (S r o) term))

end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
