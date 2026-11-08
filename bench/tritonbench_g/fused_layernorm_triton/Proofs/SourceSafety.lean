import bench.tritonbench_g.fused_layernorm_triton.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety
import bench.tritonbench_g.fused_layernorm_triton.Proofs.ReadableSafety

/-! The actual source access walk reuses the established Welford and
normalization execution contexts. Only active source addresses need backing. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem source_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Contract.Configuration)
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*s.pids 0<bounds Y)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    TraceSafe (withEquality M eq32) bounds (Contract.source X W Bias Y Mean Rstd S D c) s := by
  exact ReadableSafety.source_safe_for Bool.false M eq32 s X W Bias Y Mean Rstd S D c.block c.tree
    c.positive bounds bx bw bb byOut bm br

#axiomsClean source_safe

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.SourceSafety
