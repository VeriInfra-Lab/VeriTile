import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeTagsProgram

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags
open VeriTile Triton FP.Structural
open FPStages (Stage)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem jobs_attention_dtype (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : Stage) (order : List (GridIndex stage.grid)) (s t : State α)
    (ran : runFillSchedule (order.map (fun idx=>.source (NativeLaunch.job M D stage idx))) s=some t)
    (off : Nat) :
    cellDtype (t.mem "attention" off)=
      order.foldl (fun tag idx=>stageTag stage idx.toPids off tag) (cellDtype (s.mem "attention" off)) := by
  induction order generalizing s with
  | nil =>
    simp only [List.map_nil,runFillSchedule,Option.some.injEq] at ran
    cases ran
    rfl
  | cons idx rest ih =>
    simp only [List.map_cons,runFillSchedule,FillStage.run] at ran
    cases first : (NativeLaunch.job M D stage idx).run s with
    | none => simp [first] at ran
    | some next =>
      simp only [first,Option.bind_some] at ran
      rw [ih next ran,source_attention_dtype M D stage idx s next first off]
      rfl

theorem grid_attention_dtype (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (stage : Stage) (s t : State α)
    (ran : runFillSchedule (NativeLaunch.jobs M D stage) s=some t) (off : Nat) :
    cellDtype (t.mem "attention" off)=gridTag stage off (cellDtype (s.mem "attention" off)) :=
  jobs_attention_dtype M D stage (FPGrid.jobs stage.grid) s t ran off

private theorem run_append (xs ys : List (FillStage α)) (s : State α) :
    runFillSchedule (xs++ys) s=(runFillSchedule xs s).bind (runFillSchedule ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons first rest ih =>
    simp only [List.cons_append,runFillSchedule]
    cases first.run s <;> simp [ih]

theorem producer_tag (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : Host.Dimensions) (scale : ℝ) (s t : State α)
    (ran : runFillSchedule (NativeLaunch.producer M D c scale) s=some t)
    (off : Nat) (inside : off<Host.attentionSize c) :
    cellDtype (t.mem "attention" off)=producerTag c off := by
  let zeroed := (NativeLaunch.zero M "attention" (Host.attentionSize c)).apply s
  have hz : cellDtype (zeroed.mem "attention" off)=.real := by
    simp [zeroed,NativeLaunch.zero,FiniteFill.apply,FiniteFill.Contains,inside,cellDtype]
  simp only [NativeLaunch.producer,runFillSchedule,FillStage.run,Option.bind_some] at ran
  change runFillSchedule (if c.heads*c.T=0 then [] else
    NativeLaunch.jobs M D (.inter c.producer scale) ++
      if c.K≤256 then NativeLaunch.jobs M D (.intra c.producer c.intraBK scale)
      else .fill (NativeLaunch.zero M "scratch" (Host.scratchSize c)) ::
        (NativeLaunch.jobs M D (.split c.split scale) ++ NativeLaunch.jobs M D (.merge c.split))) zeroed=some t at ran
  by_cases empty : c.heads*c.T=0
  · simp only [empty,if_true,runFillSchedule,Option.some.injEq] at ran
    rw [←ran]
    simpa [producerTag,empty] using hz
  · simp only [empty,if_false] at ran
    rw [run_append] at ran
    cases first : runFillSchedule (NativeLaunch.jobs M D (.inter c.producer scale)) zeroed with
    | none => simp [first] at ran
    | some inter =>
      simp only [first,Option.bind_some] at ran
      have hi := grid_attention_dtype M D (.inter c.producer scale) zeroed inter first off
      rw [hz] at hi
      by_cases small : c.K≤256
      · simp only [small,if_true] at ran
        rw [grid_attention_dtype M D (.intra c.producer c.intraBK scale) inter t ran off,hi]
        simp only [producerTag,empty,if_false,small,if_true]
        rfl
      · simp only [small,if_false,runFillSchedule,FillStage.run,Option.bind_some] at ran
        rw [run_append] at ran
        let scratch := (NativeLaunch.zero M "scratch" (Host.scratchSize c)).apply inter
        have hs : cellDtype (scratch.mem "attention" off)=cellDtype (inter.mem "attention" off) := by
          simp [scratch,NativeLaunch.zero,FiniteFill.apply,FiniteFill.Contains]
        cases middle : runFillSchedule (NativeLaunch.jobs M D (.split c.split scale)) scratch with
        | none => simp [scratch,middle] at ran
        | some splitResult =>
          change (runFillSchedule (NativeLaunch.jobs M D (.split c.split scale)) scratch).bind
            (runFillSchedule (NativeLaunch.jobs M D (.merge c.split)))=some t at ran
          simp only [middle,Option.bind_some] at ran
          have hm := grid_attention_dtype M D (.split c.split scale) scratch splitResult middle off
          rw [hs,hi] at hm
          rw [grid_attention_dtype M D (.merge c.split) splitResult t ran off,hm]
          simp only [producerTag,empty,if_false,small]
          rfl

theorem producer_dtype (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : Host.Dimensions) (scale : ℝ) (s t : State α)
    (ran : runFillSchedule (NativeLaunch.producer M D c scale) s=some t)
    (off : Nat) (inside : off<Host.attentionSize c) :
    cellDtype (t.mem "attention" off)=if producerExplicit c off then .fp32 else .real :=
  (producer_tag M D c scale s t ran off inside).trans (producer_tag_eq c off)

#axiomsClean producer_dtype
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeTags
