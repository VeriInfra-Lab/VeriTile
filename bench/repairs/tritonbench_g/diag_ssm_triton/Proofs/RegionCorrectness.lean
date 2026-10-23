import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardProjection
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardRun

/-! Complete regional kernel contracts. Read-only arguments are independently
bound to their views of initial memory: they need not have disjoint backing
storage. The only separation premises concern outputs. This interface does not
use FlatAlloc.Disjoint; the older finite flat contracts remain additional
results for allocations satisfying that stronger premise. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.RegionCorrectness
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

def Forward (kernel : ComputeKernel) (p : ForwardLayout) (length : Nat) : Prop :=
  p.x ≠ p.y → ∀ u : BlockState, ∃ v,
    exec kernel.toAlgKernel u = some v ∧
    (∀ (t : Fin length) (j : Fin p.B), RealReference.active u p.batch p.dim p.B j →
      v.readMem p.y (RealReference.timeOffset u p.batch p.dim p.B t.val j) =
        RealReference.diagSsmForwardSpec u p.s p.x p.l p.batch p.dim p.B t.val j) ∧
    (∀ r o, (r ≠ p.y ∨ ∀ (t : Fin length) (j : Fin p.B),
      RealReference.active u p.batch p.dim p.B j →
      o ≠ RealReference.timeOffset u p.batch p.dim p.B t.val j) →
      v.mem r o = u.mem r o)

/-- No pair of read-only inputs is required to be disjoint. -/
structure BackwardSeparation (p : BackwardLayout) : Prop where
  initial : p.gx ≠ p.s
  output : p.gx ≠ p.y
  gradient : p.gx ≠ p.gy
  transition : p.gx ≠ p.l
  initialGradient : p.gs ≠ p.gx
  transitionGradient : p.gl ≠ p.gx
  terminal : p.gl ≠ p.gs

def Backward (kernel : ComputeKernel) (p : BackwardLayout) (length : Nat) : Prop :=
  BackwardSeparation p → ∀ u : BlockState, ∃ v,
    exec kernel.toAlgKernel u = some v ∧
    (∀ (t : Fin length) (j : Fin p.B), RealReference.active u p.batch p.dim p.B j →
      v.readMem p.gx (RealReference.timeOffset u p.batch p.dim p.B t.val j) =
        RealReference.diagSsmBackwardGradXSpec u p.l p.gy p.batch p.dim p.B length t.val j) ∧
    (∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
      v.readMem p.gs (RealReference.colOffset u p.B j) =
        RealReference.diagSsmBackwardGradSAfter u p.l p.gy p.batch p.dim p.B length j length) ∧
    (∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
      v.readMem p.gl (RealReference.colOffset u p.B j) =
        RealReference.diagSsmBackwardGradLambdaAfter u p.s p.l p.y p.gy
          p.batch p.dim p.B length j length) ∧
    (∀ r o,
      (r ≠ p.gx ∨ ∀ (t : Fin length) (j : Fin p.B),
        RealReference.active u p.batch p.dim p.B j →
        o ≠ RealReference.timeOffset u p.batch p.dim p.B t.val j) →
      (r ≠ p.gs ∨ ∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
        o ≠ RealReference.colOffset u p.B j) →
      (r ≠ p.gl ∨ ∀ j : Fin p.B, RealReference.active u p.batch p.dim p.B j →
        o ≠ RealReference.colOffset u p.B j) → v.mem r o = u.mem r o)

def ComplexForward (kernel : ComputeKernel) (p : ForwardLayout) (length : Nat) : Prop :=
  ∀ u : BlockState, ∃ v, exec kernel.toAlgKernel u = some v ∧
    (∀ t, t < length → ∀ j, ComplexForwardLoop.mask p u j = Bool.true →
      ComplexMemory.read v p.y (ComplexForwardLoop.offset p u t j) =
        ComplexForwardLoop.state p u (t + 1) j) ∧
    (∀ r o, ComplexForwardLoop.Outside p u length r o → v.mem r o = u.mem r o)

def ComplexBackward (kernel : ComputeKernel) (p : BackwardLayout) (length : Nat) : Prop :=
  ComplexBackwardRun.Separation p → ∀ u : BlockState, ∃ v,
    exec kernel.toAlgKernel u = some v ∧
    (∀ t, t < length → ∀ j, ComplexBackwardLoop.mask p u j = Bool.true →
      ComplexMemory.read v p.gx (ComplexBackwardLoop.offset p u t j) =
        Reference.inputGradient length (starRingEnd ℂ (ComplexBackwardLoop.transition p u j))
          (fun q => ComplexBackwardLoop.gradOutput p u q j) t) ∧
    (∀ j, ComplexBackwardLoop.mask p u j = Bool.true →
      ComplexMemory.read v p.gs (ComplexBackwardLoop.columns p u j) =
        Reference.reverseCarry length (starRingEnd ℂ (ComplexBackwardLoop.transition p u j))
          (fun q => ComplexBackwardLoop.gradOutput p u q j) length) ∧
    (∀ j, ComplexBackwardLoop.mask p u j = Bool.true →
      ComplexMemory.read v p.gl (ComplexBackwardLoop.columns p u j) =
        Reference.transitionGradient length (starRingEnd ℂ (ComplexBackwardLoop.transition p u j))
          (starRingEnd ℂ (ComplexBackwardLoop.initial p u j))
          (fun q => starRingEnd ℂ (ComplexBackwardLoop.output p u q j))
          (fun q => ComplexBackwardLoop.gradOutput p u q j) length) ∧
    (∀ r o, ComplexBackwardRun.Outside p u length r o → v.mem r o = u.mem r o)

theorem backward_source_exec (p : BackwardLayout) (length : Nat) (u : BlockState) :
    exec (Erasure.backward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel u =
    exec (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel u := by
  unfold exec
  rw [BackwardProjection.real_source_body, RealReference.diag_ssm_backward_kernel_toAlg_body]
  simp only [RealReference.diagSsmBackwardProjectedBody, ProgramRewrite.append,
    BackwardProjection.prelude_steps]
  congr 1
  congr 1
  funext v
  simp only [stepStmts, stepForLoopAux.forLoop_unfold, stepForRangeAux.forRange_unfold]
  rw [BackwardProjection.loop_range "i" 0 length _ _
    (BackwardProjection.loop_steps p.s p.y p.gx p.gy length p.batch p.dim p.B)]

theorem forward (p : ForwardLayout) (length : Nat) :
    Forward (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType p length := by
  intro sep u
  obtain ⟨v, he, ho, hf⟩ := RealReference.dssm_runR .triv p.s p.x p.l p.y
    length p.batch p.dim p.B sep u
  refine ⟨v, ?_, ho, hf⟩
  have hb := ForwardProjection.real_source_exec p.s p.x p.l p.y length p.batch p.dim p.B u
  have he' := (execR_triv _ _).symm.trans he
  simpa only [exec, Kernels.forwardTyped, Erasure.forward_body] using hb.trans he'

theorem backward (p : BackwardLayout) (length : Nat) :
    Backward (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length := by
  intro sep u
  obtain ⟨v, he, hx, hs, hl, hf⟩ := RealReference.dssmb_runR .triv p.s p.l p.y p.gs p.gx p.gl p.gy
    length p.batch p.dim p.B sep.initial sep.output sep.gradient sep.transition
    sep.initialGradient sep.transitionGradient sep.terminal u
  refine ⟨v, ?_, hx, hs, hl, hf⟩
  have hb := backward_source_exec p length u
  have he' := (execR_triv _ _).symm.trans he
  simpa only [exec, Kernels.backwardTyped, Erasure.backward_body] using hb.trans he'

theorem complex_forward (p : ForwardLayout) (length : Nat) :
    ComplexForward (Kernels.complexForwardTyped p.s p.x p.y p.l
      length p.batch p.dim p.B).eraseDType p length := by
  intro u
  obtain ⟨v, he, hi⟩ := ComplexForwardLoop.run p u length
  refine ⟨v, ?_, hi.outputs, hi.frame⟩
  simpa only [exec, Kernels.complexForwardTyped, Erasure.complexForward_body] using he

theorem complex_backward (p : BackwardLayout) (length : Nat) :
    ComplexBackward (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length := by
  intro sep u
  obtain ⟨v, he, hx, hs, hl, hf⟩ := ComplexBackwardRun.run p u length sep
  refine ⟨v, ?_, hx, hs, hl, hf⟩
  simpa only [exec, Kernels.complexBackwardTyped, Erasure.complexBackward_body] using he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.RegionCorrectness
