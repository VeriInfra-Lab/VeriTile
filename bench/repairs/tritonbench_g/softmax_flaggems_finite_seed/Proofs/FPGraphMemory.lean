import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPLoop

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPGraphMemory
open VeriTile Triton FP.Structural FPReference FPLoop
noncomputable section
variable {α β γ : Type} [Inhabited α]

theorem scatter_congr {shape : TileShape} (Y : RegionName)
    (off : TileIndex shape → Nat) (active : TileIndex shape → Bool)
    (v w : TileIndex shape → α) (s t : State α) (hmem : s.mem=t.mem) (hv : v=w) :
    (scatter s Y off active v).mem = (scatter t Y off active w).mem := by
  subst w
  exact writeFold_mem_congr (TileShape.allIndices shape) (fun _ => Y) off
    (fun i => active i=Bool.true) (fun i => Cell.mk .real (v i)) s t hmem

theorem walk_rel (f : Nat → β → β) (g : Nat → γ → γ) (R : β → γ → Prop)
    (step : ∀ cur b c, R b c → R (f cur b) (g cur c))
    (stop stride cur : Nat) (b : β) (c : γ) (h : R b c) :
    R (walk f stop stride cur b).2 (walk g stop stride cur c).2 := by
  by_cases hz : stride=0
  · simpa [walk,hz] using h
  by_cases hc : cur<stop
  · conv => arg 1; rw [walk,if_neg hz,if_pos hc]
    conv => arg 2; rw [walk,if_neg hz,if_pos hc]
    exact walk_rel f g R step stop stride (cur+stride) (f cur b) (g cur c) (step cur b c h)
  · simpa [walk,hz,hc] using h
termination_by stop-cur

#axiomsClean scatter_congr
#axiomsClean walk_rel
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPGraphMemory
