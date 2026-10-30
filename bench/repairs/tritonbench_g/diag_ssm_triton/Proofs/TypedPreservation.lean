import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourcePair
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Full typed-state preservation for each actual copied original/candidate
pair. The older reflexivity claims against an unsigned historical model are
inapplicable once explicit source casts are retained. The independently proved
Real memory/safety bridges are in SourceReal/SourceBackward/SourceComplex files;
this module certifies the actual integer-index rewrite for every FP algebra. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.TypedPreservation
open VeriTile VeriTile.Triton Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
open scoped VeriTile.Spec
set_option autoImplicit false

def program (kernel : ComputeKernel) : FP.ProfiledRewrite.Program := ⟨kernel, []⟩
def assumptions : Spec.Assumptions FP.GuardedFragment := []

theorem forward (s x transition y : RegionName) (length batch dim block : Nat) :
    program (forwardTyped s x transition y length batch dim block) ≡[assumptions]
      program (forwardIndexCommuted s x transition y length batch dim block) :=
  by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  exact SourcePair.execution (FP.ProfiledRewrite.engine M) .forward
    {s := s, transition := transition, y := y, x := x, gradS := s, gradX := x, gradTransition := transition, gradY := y,
      length := length, batch := batch, dim := dim, block := block} u

theorem backward (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    program (backwardTyped s transition y gs gx gl gy length batch dim block) ≡[assumptions]
      program (backwardIndexCommuted s transition y gs gx gl gy length batch dim block) :=
  by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  exact SourcePair.execution (FP.ProfiledRewrite.engine M) .backward
    {s := s, transition := transition, y := y, x := s, gradS := gs, gradX := gx, gradTransition := gl, gradY := gy,
      length := length, batch := batch, dim := dim, block := block} u

theorem complex_forward (s x y transition : RegionName) (length batch dim block : Nat) :
    program (complexForwardTyped s x y transition length batch dim block) ≡[assumptions]
      program (complexForwardIndexCommuted s x y transition length batch dim block) :=
  by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  exact SourcePair.execution (FP.ProfiledRewrite.engine M) .complexForward
    {s := s, transition := transition, y := y, x := x, gradS := s, gradX := x, gradTransition := transition, gradY := y,
      length := length, batch := batch, dim := dim, block := block} u

theorem complex_backward (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    program (complexBackwardTyped s transition y gs gx gl gy length batch dim block) ≡[assumptions]
      program (complexBackwardIndexCommuted s transition y gs gx gl gy length batch dim block) :=
  by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  exact SourcePair.execution (FP.ProfiledRewrite.engine M) .complexBackward
    {s := s, transition := transition, y := y, x := s, gradS := gs, gradX := gx, gradTransition := gl, gradY := gy,
      length := length, batch := batch, dim := dim, block := block} u

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.TypedPreservation
