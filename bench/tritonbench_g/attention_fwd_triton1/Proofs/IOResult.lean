import bench.tritonbench_g.attention_fwd_triton1.Proofs.IOGeometry
import bench.tritonbench_g.attention_fwd_triton1.Proofs.ArrayAgreement

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

@[simp] theorem invocation_initial [Inhabited α] (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (head : Nat) (s : State (Term α)) :
    (invocation α c scale storeFlag ifcond head).initial s=SharedGrid.atProgram c head s := rfl

theorem grid_run [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (n : Nat) (s : State (Term α)) :
    runSchedule ((List.range n).map (invocation α c scale storeFlag ifcond)) s=
      SharedGrid.execute c scale storeFlag ifcond s n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [List.range_succ,List.map_append,List.map_cons,List.map_nil,
      runSchedule_append,runSchedule_singleton,ih,SharedGrid.execute,Invocation.run,invocation_initial]
    rfl

theorem full_run [Inhabited α] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (s : State (Term α)) :
    runSchedule (schedule α c scale storeFlag ifcond) s=SharedGrid.run c scale storeFlag ifcond s := by
  unfold schedule SharedGrid.run
  split_ifs
  · rfl
  · exact grid_run c scale storeFlag ifcond c.heads s

theorem agrees [Inhabited α] (ops : Ops α) (literal : ℝ → α) (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (values : Values α c) (s : State (Term α))
    (binding : (io ops literal c scale storeFlag ifcond).InputBinding values s)
    (head : Nat) (headBound : head<c.heads) :
    ArrayAgreement.Agree c head (Projection.inputs ops literal c
      (SharedStep.read s "q") (SharedStep.read s "k") (SharedStep.read s "v")) (logicalInputs c values) := by
  intro row col rowBound colBound
  have bound := c.main_bound head row col headBound rowBound colBound
  have q := binding (0 : Fin 3) (⟨c.mainOffset head row col,bound⟩,PUnit.unit) trivial
  have k := binding (1 : Fin 3) (⟨c.mainOffset head row col,bound⟩,PUnit.unit) trivial
  have v := binding (2 : Fin 3) (⟨c.mainOffset head row col,bound⟩,PUnit.unit) trivial
  change (s.mem "q" _).read .real=Term.input _ at q
  change (s.mem "k" _).read .real=Term.input _ at k
  change (s.mem "v" _).read .real=Term.input _ at v
  refine ⟨?_,?_,?_⟩
  · simpa [Projection.inputs,logicalInputs,SharedStep.read,array,bound,Term.evaluate] using
      congrArg (fun t : Term α=>t.evaluate ops literal) q
  · simpa [Projection.inputs,logicalInputs,SharedStep.read,array,bound,Term.evaluate] using
      congrArg (fun t : Term α=>t.evaluate ops literal) k
  · simpa [Projection.inputs,logicalInputs,SharedStep.read,array,bound,Term.evaluate] using
      congrArg (fun t : Term α=>t.evaluate ops literal) v

theorem output_binding [Inhabited α] (ops : Ops α) (literal : ℝ → α) (zero : literal 0=ops.zero)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) (values : Values α c) (s : State (Term α))
    (binding : (io ops literal c scale storeFlag ifcond).InputBinding values s) :
    (io ops literal c scale storeFlag ifcond).OutputBinding values
      (reference ops literal c scale ifcond values) (SharedGrid.result c scale storeFlag ifcond s) := by
  intro o i active
  fin_cases o
  · have coord := main_coordinates c i.1.val i.1.isLt
    have live : ¬SharedGrid.Empty c := by
      unfold SharedGrid.Empty
      omega
    have output := SharedGrid.output c scale storeFlag ifcond s c.heads (mainHead c i.1.val) coord.1
      (mainRow c i.1.val) coord.2.1 ⟨mainColumn c i.1.val,coord.2.2⟩
    rw [main_offset] at output
    change mapCell (fun t=>t.evaluate ops literal)
      ((SharedGrid.result c scale storeFlag ifcond s).mem "o" i.1.val)=Cell.mk .real _
    rw [SharedGrid.result,if_neg live,output]
    change Cell.mk .real (ops.cast ((Graph.output c (SharedStep.read s "q") (SharedStep.read s "k")
      (SharedStep.read s "v") scale (mainHead c i.1.val) (mainRow c i.1.val/32) ifcond
      (⟨mainRow c i.1.val%32,Nat.mod_lt _ (by decide)⟩,⟨mainColumn c i.1.val,coord.2.2⟩,PUnit.unit)).evaluate ops literal))=_
    rw [Projection.graph_output ops literal zero]
    rw [ArrayAgreement.output ops c _ (logicalInputs c values) (literal scale)
      (mainHead c i.1.val) (mainRow c i.1.val/32) ifcond
      (agrees ops literal c scale storeFlag ifcond values s binding _ coord.1)]
    rfl
  · change storeFlag=Bool.true at active
    subst storeFlag
    have coord := history_coordinates c i.1.val i.1.isLt
    have live : ¬SharedGrid.Empty c := by
      intro empty
      rcases empty with time|heads
      · have := c.chunks_zero.mpr time; omega
      · omega
    have history := SharedGrid.history c scale ifcond s c.heads (historyHead c i.1.val) coord.1
      (historyChunk c i.1.val) coord.2.1 ⟨historyRow c i.1.val,coord.2.2.1⟩ ⟨historyColumn c i.1.val,coord.2.2.2⟩
    rw [history_offset] at history
    change mapCell (fun t=>t.evaluate ops literal)
      ((SharedGrid.result c scale Bool.true ifcond s).mem "h" i.1.val)=Cell.mk .real _
    rw [SharedGrid.result,if_neg live,history]
    change Cell.mk .real (ops.cast ((Graph.carry c (SharedStep.read s "k") (SharedStep.read s "v")
      (historyHead c i.1.val) ifcond (historyChunk c i.1.val)
      (⟨historyRow c i.1.val,coord.2.2.1⟩,⟨historyColumn c i.1.val,coord.2.2.2⟩,PUnit.unit)).evaluate ops literal))=_
    rw [Projection.graph_carry ops literal zero c (SharedStep.read s "q") _ _ scale]
    rw [ArrayAgreement.carry ops c _ (logicalInputs c values) (literal scale)
      (historyHead c i.1.val) ifcond (historyChunk c i.1.val)
      (agrees ops literal c scale Bool.true ifcond values s binding _ coord.1)]
    rfl

theorem frame [Inhabited α] (ops : Ops α) (literal : ℝ → α) (c : Shape) (scale : ℝ)
    (storeFlag ifcond : Bool) (values : Values α c) (s : State (Term α)) :
    (io ops literal c scale storeFlag ifcond).Frame values s (SharedGrid.result c scale storeFlag ifcond s) := by
  intro r off outside
  unfold SharedGrid.result
  split_ifs with empty
  · rfl
  · apply SharedGrid.frame c scale storeFlag ifcond s c.heads (Nat.le_refl _) r off
    · by_cases region : r="o"
      · right
        by_contra bound
        have hit := outside.1 (0 : Fin 2) (⟨off,by omega⟩,PUnit.unit) trivial
        exact hit.elim (fun ne=>ne region) (fun ne=>ne rfl)
      · exact Or.inl region
    · cases hs : storeFlag
      · exact Or.inl rfl
      · right
        by_cases region : r="h"
        · right
          by_contra bound
          have hit := outside.1 (1 : Fin 2) (⟨off,by omega⟩,PUnit.unit) hs
          exact hit.elim (fun ne=>ne region) (fun ne=>ne rfl)
        · exact Or.inl region

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.KernelIO
