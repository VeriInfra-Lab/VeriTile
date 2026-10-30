import bench.tritonbench_g.fused_recurrent_delta.MathematicalIO
import bench.tritonbench_g.fused_recurrent_delta.Proofs.MathAgreement
import bench.tritonbench_g.fused_recurrent_delta.Proofs.IOResult
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealSource
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IORealForward
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters Interface
noncomputable section
set_option maxHeartbeats 3000000

theorem primal_agrees (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : ForwardInputs ℝ s) (u : State ℝ)
    (hin : (forwardIO KernelReal.algebra s f scale bh iv).InputBinding vals u)
    (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
    MathAgreement.Agree s.T (KernelReal.unpaddedColumn s f bh (iv*s.FBV+j.val) u) (primal s f vals j) := by
  constructor
  · intro t ht
    funext i
    have read := hin (0 : Fin 6) (⟨t,ht⟩,wideKey s i,()) i.isLt
    simpa [KernelReal.unpaddedColumn,primal,ht,forwardIO,forwardInputs,wideKey,mapValue] using read
  · intro t ht
    funext i
    have read := hin (1 : Fin 6) (⟨t,ht⟩,wideKey s i,()) i.isLt
    simpa [KernelReal.unpaddedColumn,primal,ht,forwardIO,forwardInputs,wideKey,mapValue] using read
  · intro t ht
    have read := hin (2 : Fin 6) (⟨t,ht⟩,j,()) active
    simpa [KernelReal.unpaddedColumn,primal,ht,forwardIO,forwardInputs,valAddress,Nat.add_assoc,mapValue] using read
  · intro t ht
    cases flag : f.vectorBeta
    · have read := hin (4 : Fin 6) (⟨t,ht⟩,()) flag
      simpa [KernelReal.unpaddedColumn,primal,ht,flag,forwardIO,forwardInputs,mapValue] using read
    · have read := hin (3 : Fin 6) (⟨t,ht⟩,j,()) ⟨flag,active⟩
      simpa [KernelReal.unpaddedColumn,primal,ht,flag,forwardIO,forwardInputs,valAddress,Nat.add_assoc,mapValue] using read
  · funext i
    cases flag : f.initial
    · simp [KernelReal.unpaddedColumn,primal,flag]
    · have read := hin (5 : Fin 6) (wideKey s i,j,()) ⟨flag,i.isLt,active⟩
      simpa [KernelReal.unpaddedColumn,primal,flag,forwardIO,forwardInputs,wideKey,stateAddress,Nat.add_assoc,mapValue] using read

theorem complete (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    (forwardIO KernelReal.algebra s f scale bh iv).Implements (forwardFormula s f scale) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact IOSafety.forward_safe KernelReal.algebra s f scale bh iv vals bounds hb initial
  · intro vals initial hin
    let start := (job KernelReal.algebra .forward s f scale bh iv).initial initial
    have bound : (forwardIO KernelReal.algebra s f scale bh iv).InputBinding vals start := by
      intro port j active
      simpa only [start,Invocation.initial_mem] using hin port j active
    have pid := IOSafety.job_pids KernelReal.algebra .forward s f scale bh iv initial
    obtain ⟨final,facts,values⟩ := KernelReal.forward_complete s f scale bh iv start pid.1 pid.2.1 pid.2.2
    refine ⟨final,?_,?_,?_⟩
    · simpa only [forwardIO,runSchedule_singleton,Invocation.run,job] using facts.success
    · rintro ⟨port,hp⟩
      match port with
      | 0 =>
        intro j active
        have agree := primal_agrees s f scale bh iv vals start bound j.2.1 active
        have value := values _ _ _ (KernelReal.ForwardExpected.output j.1 j.2.1 active)
        rw [facts.outputs j.1 j.2.1 active,KernelReal.stored_real,agree.output scale j.1.val j.1.isLt] at value
        change mapCell id (final.mem "o" (valAddress s bh iv s.FBV j.1.val j.2.1.val))=
          Cell.mk .real (Math.output scale (primal s f vals j.2.1) j.1.val)
        rw [facts.outputs j.1 j.2.1 active]
        exact congrArg (Cell.mk .real) value
      | 1 =>
        intro j active
        have agree := primal_agrees s f scale bh iv vals start bound j.2.1 active
        have value := values _ _ _ (KernelReal.ForwardExpected.residual j.1 j.2.1 active)
        rw [facts.residuals j.1 j.2.1 active,KernelReal.stored_real,agree.residual j.1.val j.1.isLt] at value
        change mapCell id (final.mem "v" (valAddress s bh iv s.FBV j.1.val j.2.1.val))=
          Cell.mk .real (Math.residual (Math.state (primal s f vals j.2.1) j.1.val)
            ((primal s f vals j.2.1).k j.1.val) ((primal s f vals j.2.1).x j.1.val))
        rw [facts.residuals j.1 j.2.1 active]
        exact congrArg (Cell.mk .real) value
      | 2 =>
        intro j active
        change f.final=Bool.true ∧ j.1.val<s.K ∧ iv*s.FBV+j.2.1.val<s.V at active
        have agree := primal_agrees s f scale bh iv vals start bound j.2.1 active.2.2
        have value := values _ _ _ (KernelReal.ForwardExpected.terminal active.1 ⟨j.1.val,active.2.1⟩ j.2.1 active.2.2)
        rw [facts.terminal active.1 j.1 j.2.1 active.2.1 active.2.2,KernelReal.stored_real,
          agree.state s.T le_rfl] at value
        change mapCell id (final.mem "ht" (stateAddress s bh iv s.FBV j.1.val j.2.1.val))=
          Cell.mk .real (if hi : j.1.val<s.K then Math.state (primal s f vals j.2.1) s.T ⟨j.1.val,hi⟩ else 0)
        rw [facts.terminal active.1 j.1 j.2.1 active.2.1 active.2.2,dif_pos active.2.1]
        exact congrArg (Cell.mk .real) value
      | n+3 => simp [forwardIO,forwardOutputs] at hp
    · intro r off outside
      exact (facts.frame r off (IOFrame.forward KernelReal.algebra s f scale bh iv vals r off outside)).trans
        (congrFun (congrFun (Invocation.initial_mem _ initial) r) off)

#axiomsClean primal_agrees
#axiomsClean complete
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IORealForward
