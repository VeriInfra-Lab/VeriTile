import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.PhaseReal
import bench.repairs.tritonbench_g.lightning_attention.Proofs.TargetLoss
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.RawCorrect
open VeriTile Triton FP.Structural
noncomputable section

/-- Coordinate derivatives read from the actual raw gradient output memory. -/
def RawDerivatives (d : Host.Dimensions) (input result : Memory ℝ) : Prop :=
  ∀ head : Fin (d.B*d.H),
    (∀ t a, HasDerivAt (fun ε => Target.loss
      (Target.bump (Real.logicalQ d.parameters head.val input) t a ε)
      (Real.logicalK d.parameters head.val input) (Real.logicalV d.parameters head.val input)
      (Real.logicalG d.parameters head.val input))
      ((result "queryGradient" (head.val*d.N*d.D+t.val*d.D+a.val)).read .real) 0) ∧
    (∀ t a, HasDerivAt (fun ε => Target.loss
      (Real.logicalQ d.parameters head.val input)
      (Target.bump (Real.logicalK d.parameters head.val input) t a ε)
      (Real.logicalV d.parameters head.val input) (Real.logicalG d.parameters head.val input))
      ((result "keyGradient" (head.val*d.N*d.D+t.val*d.D+a.val)).read .real) 0) ∧
    (∀ t c, HasDerivAt (fun ε => Target.loss
      (Real.logicalQ d.parameters head.val input) (Real.logicalK d.parameters head.val input)
      (Target.bump (Real.logicalV d.parameters head.val input) t c ε)
      (Real.logicalG d.parameters head.val input))
      ((result "valueGradient" (head.val*d.N*d.E+t.val*d.E+c.val)).read .real) 0)

/-- Complete forward source grid with independent causal output, all input
regions preserved and the full exterior-cell frame. Dimensions are raw Nat
geometry and Host.Contract contains only launch/tile/index bounds. -/
theorem forward_correct (d : Host.Dimensions) (geometry : Host.Contract d) (input : Memory ℝ) :
    executeForward Real.algebra Real.dotPrimitive d input=some (Real.afterForward d input) ∧
      Real.ForwardPost d input (Real.afterForward d input) :=
  Real.forward_correct d geometry input

/-- Both backward source stages execute from the same evolving raw memory,
produce independent VJP formulas/derivatives and preserve every exterior cell. -/
theorem backward_correct (d : Host.Dimensions) (geometry : Host.Contract d) (input : Memory ℝ) :
    executeBackward Real.algebra Real.dotPrimitive d input=some (Real.backwardMemory d input) ∧
      Real.BackwardPost d input (Real.backwardMemory d input) ∧
      RawDerivatives d input (Real.backwardMemory d input) := by
  obtain ⟨run,post⟩ := Real.backward_correct d geometry input
  refine ⟨run,post,?_⟩
  intro head
  have deriv := Target.loss_coordinate_derivatives (Real.logicalQ d.parameters head.val input)
    (Real.logicalK d.parameters head.val input) (Real.logicalV d.parameters head.val input)
    (Real.logicalG d.parameters head.val input)
  exact ⟨fun t a => by rw [post.query head t a,Cell.read_same]; exact deriv.1 t a,
    fun t a => by rw [post.key head t a,Cell.read_same]; exact deriv.2.1 t a,
    fun t c => by rw [post.value head t c,Cell.read_same]; exact deriv.2.2 t c⟩

#axiomsClean forward_correct
#axiomsClean backward_correct
#specNonCircular RawDerivatives avoiding [executeForward,executeBackward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.RawCorrect
