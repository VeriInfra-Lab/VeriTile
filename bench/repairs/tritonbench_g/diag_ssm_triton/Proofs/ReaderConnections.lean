import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.DiagSsmTriton.ReaderConnections
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton Interface
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Recurrence uniqueness transports the complete regional forward judgment. -/
theorem forwardReal (p : ForwardLayout) (length : Nat) (sep : p.x≠p.y)
    (state : ℝ → ℝ → (Nat → ℝ) → Nat → ℝ)
    (zero : ∀ initial transition input,state initial transition input 0=initial)
    (step : ∀ initial transition input t,state initial transition input (t+1)=state initial transition input t*transition+input t) :
    LogicalIO.forwardRealView p length ⊨ (fun xs time lane =>
      state (xs.initial lane) (xs.transition lane) (fun t => LogicalIO.timeAt xs.input t lane) (time.val+1)) := by
  have formula : (fun (xs : LogicalIO.ForwardInputs ℝ p.B length) time lane =>
      state (xs.initial lane) (xs.transition lane) (fun t => LogicalIO.timeAt xs.input t lane) (time.val+1)) = LogicalIO.forwardMath := by
    funext xs time lane
    exact HeadlineRecurrences.state_eq _ _ _ _ (zero _ _ _) (step _ _ _) _
  exact formula ▸ LogicalIO.forwardReal_correct p length sep

/-- Recurrence uniqueness transports the complete regional forward judgment. -/
theorem forwardComplex (p : ForwardLayout) (length : Nat) 
    (state : ℂ → ℂ → (Nat → ℂ) → Nat → ℂ)
    (zero : ∀ initial transition input,state initial transition input 0=initial)
    (step : ∀ initial transition input t,state initial transition input (t+1)=state initial transition input t*transition+input t) :
    LogicalIO.forwardComplexView p length ⊨ (fun xs time lane =>
      state (xs.initial lane) (xs.transition lane) (fun t => LogicalIO.timeAt xs.input t lane) (time.val+1)) := by
  have formula : (fun (xs : LogicalIO.ForwardInputs ℂ p.B length) time lane =>
      state (xs.initial lane) (xs.transition lane) (fun t => LogicalIO.timeAt xs.input t lane) (time.val+1)) = LogicalIO.forwardMath := by
    funext xs time lane
    exact HeadlineRecurrences.state_eq _ _ _ _ (zero _ _ _) (step _ _ _) _
  exact formula ▸ LogicalIO.forwardComplex_correct p length

/-- The full backward judgment follows from the displayed reverse recurrences. -/
theorem backwardReal (p : BackwardLayout) (length : Nat) (sep : RegionCorrectness.BackwardSeparation p)
    (carry : Nat → ℝ → (Nat → ℝ) → Nat → ℝ)
    (acc : Nat → ℝ → ℝ → (Nat → ℝ) → (Nat → ℝ) → Nat → ℝ)
    (carryZero : ∀ length transition gy,carry length transition gy 0=0)
    (carryStep : ∀ length transition gy k,carry length transition gy (k+1)=
      (gy (length-1-k)+carry length transition gy k)*transition)
    (accZero : ∀ length transition initial saved gy,acc length transition initial saved gy 0=0)
    (accStep : ∀ length transition initial saved gy k,acc length transition initial saved gy (k+1)=
      acc length transition initial saved gy k+(gy (length-1-k)+carry length transition gy k)*
        (if length-1-k=0 then initial else saved (length-1-k-1))) :
    LogicalIO.backwardRealView p length ⊨ (fun xs =>
      LogicalIO.BackwardOutputs.mk
        (fun time lane =>
          let transition := id (xs.transition lane)
          let gy := fun t => LogicalIO.timeAt xs.upstream t lane
          gy time.val+carry length transition gy (length-1-time.val))
        (fun lane => carry length (id (xs.transition lane)) (fun t => LogicalIO.timeAt xs.upstream t lane) length)
        (fun lane => acc length (id (xs.transition lane)) (id (xs.initial lane))
          (fun t => id (LogicalIO.timeAt xs.saved t lane)) (fun t => LogicalIO.timeAt xs.upstream t lane) length)) := by
  have carried (transition : ℝ) (gy : Nat → ℝ) (k : Nat) :
      carry length transition gy k=Reference.reverseCarry length transition gy k :=
    HeadlineRecurrences.carry_eq _ _ _ _ (carryZero _ _ _) (carryStep _ _ _) _
  have summed (transition initial : ℝ) (saved gy : Nat → ℝ) (k : Nat) :
      acc length transition initial saved gy k=Reference.transitionGradient length transition initial saved gy k :=
    HeadlineRecurrences.transitionSum_eq _ _ _ _ _ _ _ (carried transition gy) (accZero _ _ _ _ _) (accStep _ _ _ _ _) _
  simpa only [LogicalIO.backwardMath,Reference.inputGradient,carried,summed] using
    LogicalIO.backwardReal_correct p length sep

/-- The full backward judgment follows from the displayed reverse recurrences. -/
theorem backwardComplex (p : BackwardLayout) (length : Nat) (sep : ComplexBackwardRun.Separation p)
    (carry : Nat → ℂ → (Nat → ℂ) → Nat → ℂ)
    (acc : Nat → ℂ → ℂ → (Nat → ℂ) → (Nat → ℂ) → Nat → ℂ)
    (carryZero : ∀ length transition gy,carry length transition gy 0=0)
    (carryStep : ∀ length transition gy k,carry length transition gy (k+1)=
      (gy (length-1-k)+carry length transition gy k)*transition)
    (accZero : ∀ length transition initial saved gy,acc length transition initial saved gy 0=0)
    (accStep : ∀ length transition initial saved gy k,acc length transition initial saved gy (k+1)=
      acc length transition initial saved gy k+(gy (length-1-k)+carry length transition gy k)*
        (if length-1-k=0 then initial else saved (length-1-k-1))) :
    LogicalIO.backwardComplexView p length ⊨ (fun xs =>
      LogicalIO.BackwardOutputs.mk
        (fun time lane =>
          let transition := (starRingEnd ℂ) (xs.transition lane)
          let gy := fun t => LogicalIO.timeAt xs.upstream t lane
          gy time.val+carry length transition gy (length-1-time.val))
        (fun lane => carry length ((starRingEnd ℂ) (xs.transition lane)) (fun t => LogicalIO.timeAt xs.upstream t lane) length)
        (fun lane => acc length ((starRingEnd ℂ) (xs.transition lane)) ((starRingEnd ℂ) (xs.initial lane))
          (fun t => (starRingEnd ℂ) (LogicalIO.timeAt xs.saved t lane)) (fun t => LogicalIO.timeAt xs.upstream t lane) length)) := by
  have carried (transition : ℂ) (gy : Nat → ℂ) (k : Nat) :
      carry length transition gy k=Reference.reverseCarry length transition gy k :=
    HeadlineRecurrences.carry_eq _ _ _ _ (carryZero _ _ _) (carryStep _ _ _) _
  have summed (transition initial : ℂ) (saved gy : Nat → ℂ) (k : Nat) :
      acc length transition initial saved gy k=Reference.transitionGradient length transition initial saved gy k :=
    HeadlineRecurrences.transitionSum_eq _ _ _ _ _ _ _ (carried transition gy) (accZero _ _ _ _ _) (accStep _ _ _ _ _) _
  simpa only [LogicalIO.backwardMath,Reference.inputGradient,carried,summed] using
    LogicalIO.backwardComplex_correct p length sep

end
end VeriTile.Bench.Repairs.DiagSsmTriton.ReaderConnections
