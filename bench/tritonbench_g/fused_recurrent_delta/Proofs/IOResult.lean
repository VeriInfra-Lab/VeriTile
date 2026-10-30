import bench.tritonbench_g.fused_recurrent_delta.Proofs.IOInput
import bench.tritonbench_g.fused_recurrent_delta.Proofs.IOFrame
import bench.tritonbench_g.fused_recurrent_delta.Proofs.IOSafety
import bench.tritonbench_g.fused_recurrent_delta.Proofs.LogicalMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOResult
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters Interface
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    (forwardIO M s f scale bh iv).Implements (forwardReference M s f scale iv) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact IOSafety.forward_safe M s f scale bh iv vals bounds hb initial
  · intro vals initial hin
    let start := (job M .forward s f scale bh iv).initial initial
    have bound : (forwardIO M s f scale bh iv).InputBinding vals start := by
      intro port j active
      simpa only [start,Invocation.initial_mem] using hin port j active
    have agree := IOInput.forward_data M s f scale bh iv vals start bound
    have pid := IOSafety.job_pids M .forward s f scale bh iv initial
    obtain ⟨final,facts⟩ := KernelForward.complete M s f scale bh iv start pid.1 pid.2.1 pid.2.2
    refine ⟨final,?_,?_,?_⟩
    · simpa only [forwardIO,runSchedule_singleton,Invocation.run,job] using facts.success
    · rintro ⟨port,hp⟩
      match port with
      | 0 =>
        intro j active
        change mapCell id (final.mem "o" (valAddress s bh iv s.FBV j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelReference.output M s (forwardData M s f iv vals) scale j.1.val j.2.1))
        rw [facts.outputs j.1 j.2.1 active,agree.output M s scale j.1.val j.1.isLt]
        rfl
      | 1 =>
        intro j active
        change mapCell id (final.mem "v" (valAddress s bh iv s.FBV j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelReference.residual M s (forwardData M s f iv vals) j.1.val j.2.1))
        rw [facts.residuals j.1 j.2.1 active,agree.residual M s j.1.val j.1.isLt]
        rfl
      | 2 =>
        intro j active
        change f.final=Bool.true ∧ j.1.val<s.K ∧ iv*s.FBV+j.2.1.val<s.V at active
        change mapCell id (final.mem "ht" (stateAddress s bh iv s.FBV j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelReference.forward M s (forwardData M s f iv vals) s.T (j.2.1,j.1,())))
        rw [facts.terminal active.1 j.1 j.2.1 active.2.1 active.2.2]
        unfold KernelForwardLoop.carry
        rw [agree.carry M s s.T le_rfl]
        rfl
      | n+3 => simp [forwardIO,forwardOutputs] at hp
    · intro r off outside
      exact (facts.frame r off (IOFrame.forward M s f scale bh iv vals r off outside)).trans
        (congrFun (congrFun (Invocation.initial_mem _ initial) r) off)

theorem backward (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    (backwardIO M s f scale bh iv).Implements (backwardReference M s f scale iv) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact IOSafety.backward_safe M s f scale bh iv vals bounds hb initial
  · intro vals initial hin
    let start := (job M .backward s f scale bh iv).initial initial
    have bound : (backwardIO M s f scale bh iv).InputBinding vals start := by
      intro port j active
      simpa only [start,Invocation.initial_mem] using hin port j active
    have agree := IOInput.backward_data M s f scale bh iv vals start bound
    have pid := IOSafety.job_pids M .backward s f scale bh iv initial
    obtain ⟨final,facts⟩ := KernelBackwardFacts.complete M s f scale bh iv start pid.1 pid.2.1 pid.2.2
    refine ⟨final,?_,?_,?_⟩
    · simpa only [backwardIO,runSchedule_singleton,Invocation.run,job] using facts.success
    · rintro ⟨port,hp⟩
      match port with
      | 0 =>
        intro j active
        change mapCell id (final.mem "dq" (keyAddress s (plane s bh iv) j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (backwardQuery M s scale (backwardData M s f iv vals) j.1.val j.2.1))
        rw [facts.query j.1 j.2.1 active,LogicalMemory.query,agree.query_value M s scale j.1.val j.1.isLt]
        rfl
      | 1 =>
        intro j active
        change mapCell id (final.mem "dk" (keyAddress s (plane s bh iv) j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (backwardKey M s scale iv (backwardData M s f iv vals) j.1.val j.2.1))
        rw [facts.key j.1 j.2.1 active,LogicalMemory.key M s f scale bh iv start j.1.val j.1.isLt,
          agree.key_value M s scale iv j.1.val j.1.isLt]
        rfl
      | 2 =>
        intro j active
        change mapCell id (final.mem "dv" (valAddress s bh iv s.BBV j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelBackwardReference.dv M s (backwardData M s f iv vals) j.1.val
            (enhanced M s scale (backwardData M s f iv vals) j.1.val) j.2.1))
        rw [facts.value j.1 j.2.1 active]
        unfold KernelBackwardFacts.value KernelBackwardFacts.enhancedAt
        rw [LogicalMemory.enhanced_time M s f scale bh iv start j.1.val j.1.isLt,
          agree.value_value M s scale j.1.val j.1.isLt]
        rfl
      | 3 =>
        intro j active
        change f.vectorBeta=Bool.true ∧ iv*s.BBV+j.2.1.val<s.V at active
        change mapCell id (final.mem "dbeta" (valAddress s (plane s bh iv) iv s.BBV j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelBackwardReference.dbeta M s (backwardData M s f iv vals) j.1.val
            (enhanced M s scale (backwardData M s f iv vals) j.1.val) j.2.1))
        rw [facts.vectorBeta active.1 j.1 j.2.1 active.2]
        unfold KernelBackwardFacts.beta KernelBackwardFacts.enhancedAt
        rw [LogicalMemory.enhanced_time M s f scale bh iv start j.1.val j.1.isLt,
          agree.beta_value M s scale j.1.val j.1.isLt]
        rfl
      | 4 =>
        intro j active
        change mapCell id (final.mem "dbeta" (plane s bh iv*s.T+j.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelBackwardReference.betaSum M s
            (KernelBackwardReference.dbeta M s (backwardData M s f iv vals) j.1.val
              (enhanced M s scale (backwardData M s f iv vals) j.1.val))))
        rw [facts.scalarBeta active j.1]
        unfold KernelBackwardFacts.beta KernelBackwardFacts.enhancedAt
        rw [LogicalMemory.enhanced_time M s f scale bh iv start j.1.val j.1.isLt,
          agree.beta_value M s scale j.1.val j.1.isLt]
        rfl
      | 5 =>
        intro j active
        change f.initialGrad=Bool.true ∧ j.1.val<s.K ∧ iv*s.BBV+j.2.1.val<s.V at active
        change mapCell id (final.mem "dh0" (stateAddress s bh iv s.BBV j.1.val j.2.1.val))=
          Cell.mk .real (KernelReference.cast M (KernelBackwardReference.reverse M s scale (backwardData M s f iv vals) s.T j))
        rw [facts.seed active.1 j.1 j.2.1 active.2.1 active.2.2]
        unfold KernelReverseMemory.carry
        rw [agree.reverse M s scale s.T le_rfl]
        cases j with | mk i j => cases j with | mk j tail => cases tail; rfl
      | n+6 => simp [backwardIO,backwardOutputs] at hp
    · intro r off outside
      exact (facts.frame r off (IOFrame.backward M s f scale bh iv vals r off outside)).trans
        (congrFun (congrFun (Invocation.initial_mem _ initial) r) off)

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOResult
