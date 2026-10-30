import bench.tritonbench_g.kldiv_ops.Proofs.FP

namespace VeriTile.Bench.Repairs.KldivOps.TypedPreservation
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open VeriTile.Bench.TritonBenchG.KldivOps TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def defaultProgram (v : Bool) (T O : RegionName) (cols block : Nat) : Program where
  kernel := backwardDefault .fp32 .fp32 v T O cols cols cols block
  domain := []
def logProgram (v : Bool) (T O : RegionName) (cols block : Nat) : Program where
  kernel := backwardLog .fp32 .fp32 v T O cols cols cols block
  domain := []

/-- Reuse the unchanged complete backward body's signed pointer-product
commutation. This is a body-preservation result, not repaired/original wrapper
 equivalence and not a claim about a native numeric primitive. -/
specification default_body (T O : RegionName) (cols block : Nat) :
    defaultProgram Bool.false T O cols block ≡[assumptions] defaultProgram Bool.true T O cols block := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl (by rfl)
  intro α _ M D _ s _
  exact FPProof.backward_default_equivalence M .fp32 .fp32 T O cols cols cols block s

specification log_body (T O : RegionName) (cols block : Nat) :
    logProgram Bool.false T O cols block ≡[assumptions] logProgram Bool.true T O cols block := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl (by rfl)
  intro α _ M D _ s _
  exact FPProof.backward_log_equivalence M .fp32 .fp32 T O cols cols cols block s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivOps.TypedPreservation
