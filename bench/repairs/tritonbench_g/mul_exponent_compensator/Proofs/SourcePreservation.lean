import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Commuted
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.LegacySource

/-! The original integer casts now produce signed mathematical registers.
They are not AST-identical to the old Nat registers. Connect both complete
executions by their full memory result, for all sizes, states and FP callbacks.
The fixed-width index bounds retain their separate original proof. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.SourcePreservation
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem original {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ actual previous,
      FP.Structural.exec M (candidateKernel p.x p.output p.n p.block) s = some actual ∧
      FP.Structural.exec M (LegacyBody.candidateKernel p.x p.output p.n p.block) s = some previous ∧
      actual.mem = previous.mem := by
  obtain ⟨a,ha,ma⟩ := complete_correct M p s
  obtain ⟨b,hb,mb⟩ := LegacyBody.complete_correct M p s
  exact ⟨a,b,ha,hb,ma.trans mb.symm⟩

private theorem previous_scalar_left_execution {α : Type} [Inhabited α]
    (M : Algebra α) (src dst : RegionName) (n B : Nat) (s : State α) :
    FP.Structural.exec M (LegacyBody.scalarLeftKernel src dst n B) s =
      FP.Structural.exec (Commuted.flipMul M) (LegacyBody.candidateKernel src dst n B) s := by
  simp [LegacyBody.scalarLeftKernel,LegacyBody.candidateKernel,
    FP.Structural.exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numericLt,natLt,bop,store,Region.cast,
    ComputeDType.eraseDType,Commuted.flipMul]
  rfl

theorem scalar_left {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    ∃ actual previous,
      FP.Structural.exec M (scalarLeftKernel p.x p.output p.n p.block) s = some actual ∧
      FP.Structural.exec M (LegacyBody.scalarLeftKernel p.x p.output p.n p.block) s = some previous ∧
      actual.mem = previous.mem := by
  obtain ⟨a,b,ha,hb,hm⟩ := original (Commuted.flipMul M) p s
  exact ⟨a,b,(Commuted.execution M p.x p.output p.n p.block s).trans ha,
    (previous_scalar_left_execution M p.x p.output p.n p.block s).trans hb,hm⟩

#axiomsClean original
#axiomsClean scalar_left
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.SourcePreservation
