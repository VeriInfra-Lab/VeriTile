import bench.repairs.tritonbench_g.swiglu_fwd.Kernels
import bench.tritonbench_g.swiglu_fwd.Proofs.FP

namespace VeriTile.Bench.Repairs.SwigluFwd.SourceEquivalence
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd (typedKernel)
open VeriTile.Bench.TritonBenchG.SwigluFwd.FPProof
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (X Y OUT : RegionName) (sx sy so N B : Nat) : Program :=
  ⟨kernel X Y OUT sx sy so N B,[]⟩
def indexCommuted (X Y OUT : RegionName) (sx sy so N B : Nat) : Program :=
  ⟨indexCommutedKernel X Y OUT sx sy so N B,[]⟩

theorem decomposition (v : Bool) (X Y OUT : RegionName) (sx sy so N B : Nat) :
    (typedKernel .fp32 .fp32 .fp32 v X Y OUT sx sy so N B).surfaceBody =
      sourcePrefix v X Y OUT sx sy so B ++
        (kernel X Y OUT sx sy so N B).surfaceBody.drop 5 := by
  cases v <;> rfl

/-- Reuse the original complete-suffix proof, with selected FP32 storage fixed.
Only integer address products change; every floating node is retained. -/
theorem equivalent (X Y OUT : RegionName) (sx sy so N B : Nat) :
    original X Y OUT sx sy so N B ≡[assumptions]
      indexCommuted X Y OUT sx sy so N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False)
    (lhs := original X Y OUT sx sy so N B) (rhs := indexCommuted X Y OUT sx sy so N B) rfl rfl
  intro α _ M D _ s _
  change run (engine M)
      (typedKernel .fp32 .fp32 .fp32 Bool.false X Y OUT sx sy so N B).surfaceBody s =
    run (engine M)
      (typedKernel .fp32 .fp32 .fp32 Bool.true X Y OUT sx sy so N B).surfaceBody s
  rw [decomposition Bool.false X Y OUT sx sy so N B,
    decomposition Bool.true X Y OUT sx sy so N B]
  exact prefix_run M s X Y OUT sx sy so B _

#print_fp_assumptions equivalent
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.SwigluFwd.SourceEquivalence
