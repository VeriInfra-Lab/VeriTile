import bench.tritonbench_g.cross_entropy1.PairIO
import bench.tritonbench_g.cross_entropy1.Proofs.SourcePair
import bench.tritonbench_g.cross_entropy1.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.CrossEntropy1.PairEquivalence
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- Both complete forward invocations succeed, are access-safe, preserve the
entire exterior frame and have equal typed output cells on unrestricted payloads. -/
theorem forward (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) (positive : 0<p.block)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss) :
    PairIO.forwardOriginal M p row block args ≡[[]]
      PairIO.forwardIndexCommuted M p row block args := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    PublicIO.forward M p row block args positive separate outputs vals trivial bounds bounded initial binding
  have sameRun : (IO.launch M p (forwardSyntax p) row block args).run initial =
      (IO.launch M p (forwardIndexCommutedSyntax p) row block args).run initial := by
    exact SourcePair.same_run (M.withDefaultPrecision .fp32) Bool.false p _
  have sameSafe : (IO.launch M p (forwardSyntax p) row block args).Safe bounds initial ↔
      (IO.launch M p (forwardIndexCommutedSyntax p) row block args).Safe bounds initial := by
    exact SourcePair.same_safe (M.withDefaultPrecision .fp32) bounds Bool.false p _
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [IO.launch M p (forwardIndexCommutedSyntax p) row block args] initial
    rw [scheduleSafe_singleton]
    apply sameSafe.mp
    exact (scheduleSafe_singleton _ _ _).mp safe
  · change runSchedule [IO.launch M p (forwardIndexCommutedSyntax p) row block args] initial=some final
    rw [runSchedule_singleton,←sameRun]
    exact (runSchedule_singleton _ _).symm.trans ran
  · intro o j _
    rfl

/-- Ignored-label upstream suppression, all active gradient lanes, aliases,
runtime scalars and the complete exterior memory frame are unchanged. -/
theorem backward (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) :
    PairIO.backwardOriginal M p row block args ≡[[]]
      PairIO.backwardIndexCommuted M p row block args := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    PublicIO.backward M p row block args vals trivial bounds bounded initial binding
  have sameRun : (IO.launch M p (backwardSyntax p) row block args).run initial =
      (IO.launch M p (backwardIndexCommutedSyntax p) row block args).run initial := by
    exact SourcePair.same_run (M.withDefaultPrecision .fp32) Bool.true p _
  have sameSafe : (IO.launch M p (backwardSyntax p) row block args).Safe bounds initial ↔
      (IO.launch M p (backwardIndexCommutedSyntax p) row block args).Safe bounds initial := by
    exact SourcePair.same_safe (M.withDefaultPrecision .fp32) bounds Bool.true p _
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [IO.launch M p (backwardIndexCommutedSyntax p) row block args] initial
    rw [scheduleSafe_singleton]
    apply sameSafe.mp
    exact (scheduleSafe_singleton _ _ _).mp safe
  · change runSchedule [IO.launch M p (backwardIndexCommutedSyntax p) row block args] initial=some final
    rw [runSchedule_singleton,←sameRun]
    exact (runSchedule_singleton _ _).symm.trans ran
  · intro o j _
    rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy1.PairEquivalence
