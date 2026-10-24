import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Frames
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.RealHost
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.RealBridge

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous.Real
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open KldivTriton.Host (World ForwardCall BackwardCall ForwardDomain BackwardDomain)
open KldivTriton.RealHost (negation arithmetic)
open scoped BigOperators
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 6000000

def adapter : Adapter ℝ ℝ := ⟨RealExecution.algebra,id,id⟩

def target {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (eps : ℝ)
    (initial : World M) (i : Nat) : ℝ :=
  RealExecution.forward call.flag eps (reduction call.mode).code call.prediction.rows call.prediction.cols
    (logical ops negation initial.memory call.prediction.view)
    (logical ops negation initial.memory call.target.view) i

def ForwardTarget {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (eps : ℝ)
    (initial : World M) (out : View) (after : World M) : Prop :=
  out.backed ∧ after.live out.storage ∧
    out.shape = (if call.mode = .none then call.prediction.view.shape else []) ∧
    ∀ i, i < out.numel → logical ops negation after.memory out i = target ops call eps initial i

theorem expected_point {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (eps : ℝ)
    (initial : World M) (mode : call.mode = .none) (i : Nat) :
    expected ops negation adapter call eps initial i = target ops call eps initial i := by
  simp only [expected,forwardExpected,mode,reduction,if_true,point,adapter,id_eq,target,
    Host.Reduction.code,RealExecution.forward,if_pos rfl]
  exact RealExecution.scalar_real _ _ _ _

theorem expected_sum {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (eps : ℝ)
    (initial : World M) (output : View) (shape : output.shape = outputShape call)
    (mode : call.mode ≠ .none) :
    (∑ i : Fin output.numel, expected ops negation adapter call eps initial i.val) =
      RealExecution.total call.flag eps call.prediction.rows call.prediction.cols
        (logical ops negation initial.memory call.prediction.view)
        (logical ops negation initial.memory call.target.view) := by
  have reduced : reduction call.mode ≠ .none := by cases h : call.mode <;> simp_all [reduction]
  have size : output.numel = call.prediction.rows := by
    simp [View.numel,shape,outputShape,Host.Reduction.outputShape,reduced]
  rw [size]
  simp only [expected,forwardExpected,reduced,if_false,row,adapter,id_eq]
  simp_rw [RealExecution.row_ideal call.flag eps call.prediction.cols (Host.blockSize call.prediction.cols)
    (Host.block_positive call.prediction.cols)]
  simp only [RealExecution.total,RealExecution.rowLoss,RealExecution.loss,
    VeriTile.Bench.TritonBenchG.KldivOps.FullMath.rowLoss]

theorem forward_target {M β : Type} (ops : Ops M ℝ β) (call : ForwardCall) (eps : ℝ)
    (initial after : World M) (output out : View) (shape : output.shape = outputShape call)
    (result : ForwardReturned ops negation arithmetic call output (expected ops negation adapter call eps initial) out after) :
    ForwardTarget ops call eps initial out after := by
  refine ⟨result.backed,result.live,?_⟩
  have value := result.value
  by_cases mode : call.mode = .none
  · simp only [mode,if_true] at value ⊢
    obtain ⟨rfl,values⟩ := value
    refine ⟨?_,?_⟩
    · simpa [outputShape,mode,reduction,Host.Reduction.outputShape,call.prediction.shape] using shape
    · intro i hi
      exact (values i hi).trans (expected_point ops call eps initial mode i)
  · simp only [mode,if_false] at value ⊢
    refine ⟨value.shape,?_⟩
    intro i hi
    have zero : i = 0 := by simpa [View.numel,value.shape] using hi
    subst i
    obtain ⟨plan,readback⟩ := value.value
    have sum := expected_sum ops call eps initial output shape mode
    have tree := KldivTriton.HostReduction.real_plan (fun i => expected ops negation adapter call eps initial i.val) plan
    rw [readback]
    change KldivTriton.Host.scalarNormalize arithmetic _
      (KldivTriton.HostReduction.eval (·+·) 0 _ plan.tree) = _
    rw [tree,sum]
    cases hm : call.mode <;> simp_all [target,RealExecution.forward,reduction,Host.Reduction.code,
      KldivTriton.Host.scalarNormalize,KldivTriton.Host.denominator,arithmetic]

theorem backward_target (flag : Bool) (t : ℝ) :
    Contiguous.derivative adapter flag t = RealExecution.backward flag t :=
  RealExecution.backward_scalar_ideal flag t

#specNonCircular target avoiding [forwardKernel,forwardExpected,expected]
#axiomsClean forward_target
#axiomsClean backward_target
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous.Real
