import bench.tritonbench_g.flash_attn.Proofs.ActualNumericExecution

namespace VeriTile.Bench.TritonBenchG.FlashAttn.ActualNumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource NumericTypedExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

private theorem sum_fin2 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

private theorem sup_fin2 {α : Type*} [SemilatticeSup α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) (h : (@Finset.univ _ I).Nonempty) :
    (@Finset.univ _ I).sup' h f = f 0 ⊔ f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  subst I
  have he : (Finset.univ : Finset (Fin 2)) = {0,1} := by decide
  simp [he]

/-- Causal masking changes both rows' denominators without skipping K/V loads. -/
theorem causal_cells :
    (post Bool.true).val.mem "O" 20 = Cell.mk .fp16 (some 0) ∧
    (post Bool.true).val.mem "L" 3 = Cell.mk .real (some 0) ∧
    (post Bool.true).val.mem "O" 25 = Cell.mk .fp16 (some 0) ∧
    (post Bool.true).val.mem "L" 4 = Cell.mk .real (some 1) := by
  rw [post_eq]
  have log_two_ne : Real.log (2 : ℝ) ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  simp [NumericTypedExecution.post,afterLoop,second,first,entry,initial,storageType,model,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,Tile.dot,
    TileShape.axisDim,TileShape.insertAxisIndex,Fin.sum_univ_succ,sum_fin2,sup_fin2,
    WithBot.realAdd,WithBot.realSub,WithBot.realMul,WithBot.realDiv,
    WithBot.realExp2,WithBot.realLog2,FP.Structural.store,State.write,
    TileShape.allIndices,List.finRange_succ,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
    two_nsmul,WithBot.coe_add_coe_eq,log_two_ne]
  all_goals norm_num [log_two_ne,WithBot.some_eq_coe,WithBot.none_eq_bot,
    ←WithBot.coe_add,Option.map₂,Option.map,Option.bind,Fintype.card_fin,two_nsmul]
  all_goals first | rfl | (repeat' constructor)

/-- The typed witness retains every raw cell outside the two actual store
footprints, including strided gaps and all input storage tags. -/
theorem raw_frame (causal : Bool) (r : RegionName) (o : Nat)
    (hO : r ≠ "O" ∨ (o ≠ 20 ∧ o ≠ 22 ∧ o ≠ 25 ∧ o ≠ 27))
    (hL : r ≠ "L" ∨ (o ≠ 3 ∧ o ≠ 4)) :
    (post causal).val.mem r o = (initial .fp16 .fp16).mem r o := by
  rw [post_eq]
  cases causal
  all_goals
    change (State.write _ "O" 27 _).mem r o = _
    rw [State.write_other _ "O" r 27 o _ (hO.imp_right (fun h => h.2.2.2))]
    change (State.write _ "O" 25 _).mem r o = _
    rw [State.write_other _ "O" r 25 o _ (hO.imp_right (fun h => h.2.2.1))]
    change (State.write _ "O" 22 _).mem r o = _
    rw [State.write_other _ "O" r 22 o _ (hO.imp_right (fun h => h.2.1))]
    change (State.write _ "O" 20 _).mem r o = _
    rw [State.write_other _ "O" r 20 o _ (hO.imp_right (fun h => h.1))]
    change (State.write _ "L" 4 _).mem r o = _
    rw [State.write_other _ "L" r 4 o _ (hL.imp_right (fun h => h.2))]
    change (State.write _ "L" 3 _).mem r o = _
    rw [State.write_other _ "L" r 3 o _ (hL.imp_right (fun h => h.1))]
    rfl

/-- Each witness is an actual complete source execution; both source outputs
and the full raw-cell frame belong to that same successful run. -/
theorem numerical_execution (v causal : Bool) :
    ∃ t, FP.Structural.exec (engine model)
      (ActualFPProof.program v .fp16 "Q" "K" "V" "L" "O" 1
        50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal).kernel
      (initial .fp16 .fp16) = some t ∧
      t.mem "O" 20 = Cell.mk .fp16 (some (if causal then 0 else 1)) ∧
      t.mem "L" 3 = Cell.mk .real (some (if causal then 0 else 2)) ∧
      (∀ (r : RegionName) (o : Nat), (r ≠ "O" ∨ (o ≠ 20 ∧ o ≠ 22 ∧ o ≠ 25 ∧ o ≠ 27)) →
        (r ≠ "L" ∨ (o ≠ 3 ∧ o ≠ 4)) → t.mem r o = (initial .fp16 .fp16).mem r o) := by
  refine ⟨(post causal).val, ?_, ?_, ?_, raw_frame causal⟩
  · cases v
    · exact original_execution causal
    · exact candidate_execution causal
  · cases causal
    · exact noncausal_cells.1
    · exact causal_cells.1
  · cases causal
    · exact noncausal_cells.2
    · exact causal_cells.2.1

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.ActualNumericExecution
