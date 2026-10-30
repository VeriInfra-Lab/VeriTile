import bench.tritonbench_g.cross_entropy_ops.PairIO
import bench.tritonbench_g.cross_entropy_ops.Proofs.SourcePair
import bench.tritonbench_g.cross_entropy_ops.Proofs.Interface

namespace VeriTile.Bench.Repairs.CrossEntropyOps.PairEquivalence
open VeriTile.Triton FP.Structural Interface
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- All original outputs, actual source access safety and both exterior frames;
in split mode the entire z-loss view is framed. Floating payloads are unrestricted. -/
theorem forward (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) (positive : 0<p.block)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (zLse : p.zloss≠p.lse) (zLoss : p.zloss≠p.loss) :
    PairIO.forwardOriginal M p row block args ≡[[]]
      PairIO.forwardIndexCommuted M p row block args := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    Forward.implements M p args row block positive separate outputs zLse zLoss vals trivial bounds bounded initial binding
  have sameRun : (invocation M p args Bool.false row block).run initial =
      (PairIO.candidateInvocation M p row block args Bool.false).run initial := by
    exact SourcePair.same_run (KernelNative.engine M) Bool.false p _
  have sameSafe : (invocation M p args Bool.false row block).Safe bounds initial ↔
      (PairIO.candidateInvocation M p row block args Bool.false).Safe bounds initial := by
    exact SourcePair.same_safe (KernelNative.engine M) bounds Bool.false p _
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [PairIO.candidateInvocation M p row block args Bool.false] initial
    rw [scheduleSafe_singleton]
    exact sameSafe.mp ((scheduleSafe_singleton _ _ _).mp safe)
  · change runSchedule [PairIO.candidateInvocation M p row block args Bool.false] initial=some final
    rw [runSchedule_singleton,←sameRun]
    exact (runSchedule_singleton _ _).symm.trans ran
  · intro port lane _
    rfl

/-- Complete ignored-label and in-place backward behavior, with every active
lane and the complete exterior frame. Only the integer product is commuted. -/
theorem backward (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) :
    PairIO.backwardOriginal M p row block args ≡[[]]
      PairIO.backwardIndexCommuted M p row block args := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    Backward.implements M p args row block vals trivial bounds bounded initial binding
  have sameRun : (invocation M p args Bool.true row block).run initial =
      (PairIO.candidateInvocation M p row block args Bool.true).run initial := by
    exact SourcePair.same_run (KernelNative.engine M) Bool.true p _
  have sameSafe : (invocation M p args Bool.true row block).Safe bounds initial ↔
      (PairIO.candidateInvocation M p row block args Bool.true).Safe bounds initial := by
    exact SourcePair.same_safe (KernelNative.engine M) bounds Bool.true p _
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [PairIO.candidateInvocation M p row block args Bool.true] initial
    rw [scheduleSafe_singleton]
    exact sameSafe.mp ((scheduleSafe_singleton _ _ _).mp safe)
  · change runSchedule [PairIO.candidateInvocation M p row block args Bool.true] initial=some final
    rw [runSchedule_singleton,←sameRun]
    exact (runSchedule_singleton _ _).symm.trans ran
  · intro port lane _
    rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropyOps.PairEquivalence
