import bench.tritonbench_g.layer_norm_fwd.Source
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FP

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Pair
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) : Program where
  kernel := quoted f r stride M N eps p
  domain := []

def indexCommuted (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) : Program where
  kernel := Source.indexCommuted (sourceFlags f) (sourceRegions r) (sourceStrides stride)
    M N eps p (tile N) (tile_arange N)
  domain := []

/-- Reuse the already proved complete-source equality. The selected positive
tile, including N=0, every flag, cast and random stream remain in both sources. -/
theorem execution {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (s : State α) :
    exec A (original f r stride M N eps p).kernel s =
      exec A (indexCommuted f r stride M N eps p).kernel s :=
  FPProof.whole_state f r stride M N eps p (tile N) (tile_arange N) A s

theorem equivalent (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) :
    original f r stride M N eps p ≡[assumptions]
      indexCommuted f r stride M N eps p := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl (by rfl)
  intro α _ A D _ s _
  exact execution (engine A) f r stride M N eps p s

#axiomsClean execution
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Pair
