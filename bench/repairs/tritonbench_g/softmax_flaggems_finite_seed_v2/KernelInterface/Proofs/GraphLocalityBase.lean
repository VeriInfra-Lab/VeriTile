import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GridSafety
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPGraphMemory

/-! Finite read locality of the independent source-order graphs. This proof is
uniform in scalar/reduction semantics and preserves source/output aliases by
threading both memories through every emitted tile and every CTA. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α β γ : Type} [Inhabited α]

def Agree (regions : List RegionName) (n : Nat) (s t : State α) : Prop :=
  ∀ r∈regions, ∀ k,k<n → FPReference.read s r k=FPReference.read t r k

theorem write_agree (regions : List RegionName) (n : Nat) (s t : State α)
    (h : Agree regions n s t) (Y : RegionName) (offset : Nat) (v : Cell α) :
    Agree regions n (s.write Y offset v) (t.write Y offset v) := by
  intro r hr k hk
  simp only [FPReference.read,State.write]
  split
  · rfl
  · exact h r hr k hk

theorem scatter_agree {shape : TileShape} (regions : List RegionName) (n : Nat)
    (s t : State α) (h : Agree regions n s t) (Y : RegionName)
    (offset : TileIndex shape → Nat) (active : TileIndex shape → Bool)
    (v w : TileIndex shape → α) (equal : v=w) :
    Agree regions n (scatter s Y offset active v) (scatter t Y offset active w) := by
  subst w
  unfold scatter
  generalize TileShape.allIndices shape=lanes
  induction lanes generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i=Bool.true
    · simp only [hi,↓reduceIte]
      exact ih _ _ (write_agree regions n s t h Y (offset i) (.mk .real (v i)))
    · simp only [hi,↓reduceIte]
      exact ih s t h

theorem walk_rel (f : Nat → β → β) (g : Nat → γ → γ) (R : β → γ → Prop)
    (Inv : Nat → Prop) (stop stride : Nat)
    (next : ∀ cur,Inv cur → Inv (cur+stride))
    (step : ∀ cur b c,cur<stop → Inv cur → R b c → R (f cur b) (g cur c))
    (cur : Nat) (b : β) (c : γ) (hi : Inv cur) (h : R b c) :
    R (walk f stop stride cur b).2 (walk g stop stride cur c).2 := by
  by_cases hz : stride=0
  · simpa only [walk,hz,↓reduceIte] using h
  by_cases hc : cur<stop
  · conv => arg 1; rw [walk,if_neg hz,if_pos hc]
    conv => arg 2; rw [walk,if_neg hz,if_pos hc]
    exact walk_rel f g R Inv stop stride next step (cur+stride) (f cur b) (g cur c)
      (next cur hi) (step cur b c hc hi h)
  · simpa only [walk,hz,hc,↓reduceIte] using h
termination_by stop-cur

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Locality
