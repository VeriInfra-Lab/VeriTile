import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! The candidate's complete floating typed bodies are the same shared bodies
as the pinned source after erasing only the added integer casts only. check_source.py binds this statement to all four complete Python JIT
ASTs. These structural equalities are not native numerical claims. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.TypedPreservation
open VeriTile VeriTile.Triton Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
open scoped VeriTile.Spec
set_option autoImplicit false

def program (kernel : ComputeKernel) : FP.ProfiledRewrite.Program := ⟨kernel, []⟩
def assumptions : Spec.Assumptions FP.GuardedFragment := []

theorem forward (s x transition y : RegionName) (length batch dim block : Nat) :
    program (forwardTyped s x transition y length batch dim block) ≡[assumptions]
      program (TypedSource.forward .fp32 Bool.false s x transition y length batch dim block) :=
  Spec.FloatingPoint.refl _

theorem backward (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    program (backwardTyped s transition y gs gx gl gy length batch dim block) ≡[assumptions]
      program (TypedSource.backward .fp32 Bool.false s transition y gs gx gl gy length batch dim block) :=
  Spec.FloatingPoint.refl _

theorem complex_forward (s x y transition : RegionName) (length batch dim block : Nat) :
    program (complexForwardTyped s x y transition length batch dim block) ≡[assumptions]
      program (TypedSource.complexForward Bool.false s x y transition length batch dim block) :=
  Spec.FloatingPoint.refl _

theorem complex_backward (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    program (complexBackwardTyped s transition y gs gx gl gy length batch dim block) ≡[assumptions]
      program (TypedSource.complexBackward Bool.false s transition y gs gx gl gy length batch dim block) :=
  Spec.FloatingPoint.refl _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.TypedPreservation
