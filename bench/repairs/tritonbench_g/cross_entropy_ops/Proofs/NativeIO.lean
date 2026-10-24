import bench.repairs.tritonbench_g.cross_entropy_ops.NativeIO

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Interface.Native
open VeriTile.Triton FP.Structural KernelProfile KernelContract
noncomputable section
set_option maxHeartbeats 4000000

theorem forward_implements (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (row block : Nat)
    (positive : 0 < p.block) (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss) :
    (forwardIO R I S p v row block).Implements (forwardResult R I S p v row block) := by
  intro values _ bounds hb s hin
  let data := forwardInputs p row values
  have bounded : (Forward.programIO (Compiler.algebra R) p (KernelProfile.arguments v) row block).InBounds bounds data := by
    exact ⟨hb.1, hb.2.1, hb.2.2.1, fun k => Fin.elim0 k⟩
  have inputs : (Forward.programIO (Compiler.algebra R) p (KernelProfile.arguments v) row block).InputBinding data s := by
    intro i j active
    fin_cases i
    · exact hin ⟨0, by simp [forwardIO, Forward.memory, Forward.inputs]⟩ j trivial
    · exact hin ⟨1, by simp [forwardIO, Forward.memory, Forward.inputs]⟩ j trivial
  obtain ⟨safe, final, ran, observed, frame⟩ :=
    Forward.implements (Compiler.algebra R) p (KernelProfile.arguments v) row block
      positive separate outputs zLse zLoss data trivial bounds bounded s inputs
  refine ⟨safe, final, ran, ?_, ?_⟩
  rotate_left
  · intro r i outside
    exact frame r i ⟨outside.1, fun k => Fin.elim0 k⟩
  intro o j active
  have typed := observed o j active
  have observed' := congrArg (mapCell (Compiler.eval I S)) typed
  fin_cases o
  all_goals cases j
  all_goals
    change mapCell (Compiler.eval I S) (mapCell id _) = _ at observed'
    have identity (c : Cell Compiler.Term) : mapCell id c = c := by
      cases c with | mk d v => cases d <;> rfl
    rw [identity] at observed'
    exact observed'

theorem backward_implements (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (v : RuntimeArgs.Values Word) (row block : Nat) :
    (backwardIO R I S p v row block).Implements (backwardResult R I S p v row block) := by
  intro values _ bounds hb s hin
  let data := backwardInputs p row values
  have bounded : (Backward.programIO (Compiler.algebra R) p (KernelProfile.arguments v) row block).InBounds bounds data := by
    exact ⟨hb.1, hb.2.1, hb.2.2.1, fun k => Fin.elim0 k⟩
  have inputs : (Backward.programIO (Compiler.algebra R) p (KernelProfile.arguments v) row block).InputBinding data s := by
    intro i j active
    fin_cases i
    · exact hin ⟨0, by simp [backwardIO, Backward.memory, Backward.inputs]⟩ j trivial
    · exact hin ⟨1, by simp [backwardIO, Backward.memory, Backward.inputs]⟩ j trivial
    · exact hin ⟨2, by simp [backwardIO, Backward.memory, Backward.inputs]⟩ j trivial
    · exact hin ⟨3, by simp [backwardIO, Backward.memory, Backward.inputs]⟩ j trivial
  obtain ⟨safe, final, ran, observed, frame⟩ :=
    Backward.implements (Compiler.algebra R) p (KernelProfile.arguments v) row block
      data trivial bounds bounded s inputs
  refine ⟨safe, final, ran, ?_, ?_⟩
  rotate_left
  · intro r i outside
    exact frame r i ⟨outside.1, fun k => Fin.elim0 k⟩
  intro o j active
  have typed := observed o j active
  have observed' := congrArg (mapCell (Compiler.eval I S)) typed
  fin_cases o
  change mapCell (Compiler.eval I S) (mapCell id _) = _ at observed'
  have identity (c : Cell Compiler.Term) : mapCell id c = c := by
    cases c with | mk d v => cases d <;> rfl
  rw [identity] at observed'
  exact observed'

#axiomsClean forward_implements
#axiomsClean backward_implements
#specNonCircular forwardResult avoiding [forwardSyntax, backwardSyntax, exec, run]
#specNonCircular backwardResult avoiding [forwardSyntax, backwardSyntax, exec, run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Interface.Native
