import bench.tritonbench_g.layer_norm_ops.PairIO
import bench.tritonbench_g.layer_norm_ops.Proofs.SourcePair
import bench.tritonbench_g.layer_norm_ops.Proofs.IOResults

namespace VeriTile.Bench.Repairs.LayerNormOps.PairEquivalence
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- Complete forward invocations with all four flags, saved statistics,
optional residual output, arbitrary FP payloads and both exterior frames. -/
theorem forward (M : Algebra α) (p : ForwardParameters) (pid : Nat)
    (admissible : KernelExecution.ForwardAdmissible p) :
    PairIO.forwardOriginal M p pid ≡[[]] PairIO.forwardMaskReused M p pid := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    Interface.forward_implements M p pid admissible vals trivial bounds bounded initial binding
  have sameMem : ((Interface.invocation M (forwardKernel p) pid).run initial).map State.mem =
      ((Interface.invocation M (forwardMaskReusedKernel p) pid).run initial).map State.mem := by
    exact SourcePair.forward_memory M p _
  have originalRan : (Interface.invocation M (forwardKernel p) pid).run initial=some final := by
    exact (runSchedule_singleton _ _).symm.trans ran
  rw [originalRan] at sameMem
  cases hc : (Interface.invocation M (forwardMaskReusedKernel p) pid).run initial with
  | none => simp [hc] at sameMem
  | some candidate =>
    have memory : final.mem=candidate.mem := by simpa [hc] using sameMem
    refine ⟨safe,?_,final,candidate,ran,?_,?_,frame,?_⟩
    · change ScheduleSafe bounds [Interface.invocation M (forwardMaskReusedKernel p) pid] initial
      rw [scheduleSafe_singleton]
      change TraceSafe M bounds (forwardMaskReusedKernel p)
        ((Interface.invocation M (forwardMaskReusedKernel p) pid).initial initial)
      apply SourcePair.forward_candidate_safe
      · intro j hj
        exact bounded.2.1 ⟨0,by simp [Interface.forwardIO,Interface.forwardInputs]⟩ (j,()) hj
      · intro j hj
        exact bounded.2.1 ⟨1,by simp [Interface.forwardIO,Interface.forwardInputs]⟩ (j,()) hj
      · intro h j hj
        exact bounded.2.1 ⟨2,by simp [Interface.forwardIO,Interface.forwardInputs]⟩ (j,()) ⟨h,hj⟩
      · intro h j hj
        exact bounded.2.1 ⟨3,by simp [Interface.forwardIO,Interface.forwardInputs]⟩ (j,()) ⟨h,hj⟩
      · intro j hj
        exact bounded.2.2.1 ⟨0,by simp [Interface.forwardIO,Interface.forwardOutputs]⟩ (j,()) hj
      · intro h j hj
        exact bounded.2.2.1 ⟨1,by simp [Interface.forwardIO,Interface.forwardOutputs]⟩ (j,()) ⟨h,hj⟩
      · intro h
        exact bounded.2.2.1 ⟨2,by simp [Interface.forwardIO,Interface.forwardOutputs]⟩ () h
      · exact bounded.2.2.1 ⟨3,by simp [Interface.forwardIO,Interface.forwardOutputs]⟩ () trivial
    · change runSchedule [Interface.invocation M (forwardMaskReusedKernel p) pid] initial=some candidate
      rw [runSchedule_singleton,hc]
    · intro o j _
      change mapCell id (final.mem _ _) = mapCell id (candidate.mem _ _)
      rw [memory]
    · change (Interface.forwardIO M p pid).Frame vals initial candidate
      unfold ProgramIO.Frame at frame ⊢
      simpa only [←memory] using frame

/-- Complete bounded backward loop, including empty loops and every optional
recompute/residual/parameter-gradient output. Only pure integer work changes. -/
theorem backward (M : Algebra α) (p : BackwardParameters) (pid : Nat)
    (admissible : KernelReadback.Admissible p) :
    PairIO.backwardOriginal M p pid ≡[[]] PairIO.backwardRowStartReused M p pid := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro vals _ bounds bounded initial binding
  obtain ⟨safe,final,ran,values,frame⟩ :=
    Interface.backward_implements M p pid admissible vals trivial bounds bounded initial binding
  have sameRun : (Interface.invocation M (backwardKernel p) pid).run initial =
      (Interface.invocation M (backwardRowStartReusedKernel p) pid).run initial := by
    exact SourcePair.backward_run M p _
  have sameSafe : (Interface.invocation M (backwardKernel p) pid).Safe bounds initial ↔
      (Interface.invocation M (backwardRowStartReusedKernel p) pid).Safe bounds initial := by
    exact SourcePair.backward_safe M bounds p _
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · change ScheduleSafe bounds [Interface.invocation M (backwardRowStartReusedKernel p) pid] initial
    rw [scheduleSafe_singleton]
    exact sameSafe.mp ((scheduleSafe_singleton _ _ _).mp safe)
  · change runSchedule [Interface.invocation M (backwardRowStartReusedKernel p) pid] initial=some final
    rw [runSchedule_singleton,←sameRun]
    exact (runSchedule_singleton _ _).symm.trans ran
  · intro o j _
    rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.LayerNormOps.PairEquivalence
