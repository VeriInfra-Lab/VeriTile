import bench.tritonbench_g.bgmv_expand_slice.SignedReadback
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Geometry
import bench.tritonbench_g.bgmv_expand_slice.SignedControl
import bench.tritonbench_g.bgmv_expand_slice.SignedStages

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Execution
open VeriTile.Triton FP.Structural Program Direct Direct.FPValues Setup Body Readback Stages
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop_executes (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) (positiveN : 0<h.N) :
    step (machine M reduction) (Program.loop h) (readyState M reduction h batch row slot data metadata)=
      some (finalState M reduction h batch row slot data metadata) := by
  exact Control.loop_one (machine M reduction) (body h)
    (readyState M reduction h batch row slot data metadata)
    (finalState M reduction h batch row slot data metadata)
    (ready_length M reduction h batch row slot data metadata positiveN)
    (body_runs M reduction h batch row slot data metadata)

#axiomsClean loop_executes

theorem active_executes (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) (positiveN : 0<h.N) :
    run (machine M reduction) (active h) (preguard M reduction h batch row data).val=
      some (finalState M reduction h batch row slot data metadata) := by
  rw [complete_active,run_append,ready_runs M reduction h batch row slot data metadata,Option.bind_some]
  exact Control.single _ _ _ _ (loop_executes M reduction h batch row slot data metadata positiveN)

#axiomsClean active_executes

theorem condition_eval (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row : Nat) (data : Data α) :
    FP.Structural.evalOp (machine M reduction) none guard (preguard M reduction h batch row data).val=
      some (fun _ => decide (data.indices batch≠-1)) := by
  apply Control.integer_guard
  rfl


#axiomsClean condition_eval

/-- Complete typed-source execution and exact tagged memory. All numerical
operations remain the actual source's local casts, product, reduction and add. -/
theorem source_executes (M : Algebra α) (reduction : Reduction α) (h : Call)
    (p : Geometry.Program h) (data : Data α)
    (valid : data.indices p.batch=-1 ∨ 0≤data.indices p.batch) :
    ∃ t, FP.Structural.exec (machine M reduction) (Surface.source h) (entry h p.batch p.row data)=some t ∧
      t.mem=(if data.indices p.batch=-1 then (entry h p.batch p.row data).mem else
        ((entry h p.batch p.row data).write "Y" (p.batch*h.columns+p.row+h.slice)
          (.mk .real (result M reduction h p.batch (data.indices p.batch).toNat p.row data ⟨0,by decide⟩))).mem) := by
  by_cases sentinel : data.indices p.batch=-1
  · refine ⟨(preguard M reduction h p.batch p.row data).val,?_,?_⟩
    · change run (machine M reduction) (code h) (entry h p.batch p.row data)=_
      rw [complete_code,run_append]
      simp only [(preguard M reduction h p.batch p.row data).property,Option.bind_some]
      apply Control.single
      apply Control.skip
      simpa only [sentinel,ne_self_iff_false,decide_false] using condition_eval M reduction h p.batch p.row data
    · simp only [sentinel,if_true]
      rfl
  · have nonnegative := valid.resolve_left sentinel
    have metadata : data.indices p.batch=((data.indices p.batch).toNat:Int) := (Int.toNat_of_nonneg nonnegative).symm
    have positiveN : 0<h.N := by have := p.rowInside; omega
    refine ⟨(finalState M reduction h p.batch p.row (data.indices p.batch).toNat data metadata),?_,?_⟩
    · change run (machine M reduction) (code h) (entry h p.batch p.row data)=_
      rw [complete_code,run_append]
      simp only [(preguard M reduction h p.batch p.row data).property,Option.bind_some]
      apply Control.single
      apply Control.pass
      · simpa [sentinel] using condition_eval M reduction h p.batch p.row data
      · exact active_executes M reduction h p.batch p.row (data.indices p.batch).toNat data metadata positiveN
    · simpa only [sentinel,if_false] using final_memory M reduction h p.batch p.row
        (data.indices p.batch).toNat data metadata positiveN

#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Execution
