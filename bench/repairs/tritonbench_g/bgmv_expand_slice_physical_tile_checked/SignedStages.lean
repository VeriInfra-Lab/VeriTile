import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedReadback
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.Geometry
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedControl

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Stages
open VeriTile.Triton FP.Structural Program Direct.FPValues Setup Body Readback
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem preloop_length (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) (positiveN : 0<h.N) :
    (preloop M reduction h batch row slot data metadata).val.regs .int [] "split_n_length"=
      some (fun _ => 1) := by
  have length := signed_length h positiveN
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  have positive_sum : ¬((N:Int)+N<1) := by change 0<N at positiveN;omega
  cases ev
  all_goals
    simp [preloop,integralFloorDiv]
    funext i
    simpa only [if_neg positive_sum] using length


@[irreducible] def readyState (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) : State α :=
  (preloop M reduction h batch row slot data metadata).val

@[irreducible] def finalState (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) : State α :=
  (iteration M reduction h batch row slot data metadata).val

theorem ready_runs (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    run (machine M reduction) (setup h) (preguard M reduction h batch row data).val=
      some (readyState M reduction h batch row slot data metadata) := by
  unfold readyState
  exact (preloop M reduction h batch row slot data metadata).property

theorem ready_length (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) (positiveN : 0<h.N) :
    (readyState M reduction h batch row slot data metadata).regs .int [] "split_n_length"=some (fun _ => 1) := by
  unfold readyState
  exact preloop_length M reduction h batch row slot data metadata positiveN

theorem body_runs (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    run (machine M reduction) (body h) ((readyState M reduction h batch row slot data metadata).setReg "n" .int [] (fun _ => 0))=
      some (finalState M reduction h batch row slot data metadata) := by
  unfold readyState finalState
  exact (iteration M reduction h batch row slot data metadata).property

theorem final_memory (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) (positiveN : 0<h.N) :
    (finalState M reduction h batch row slot data metadata).mem=
      ((entry h batch row data).write "Y" (batch*h.columns+row+h.slice)
        (.mk .real (result M reduction h batch slot row data ⟨0,by decide⟩))).mem := by
  unfold finalState
  exact iteration_memory M reduction h batch row slot data metadata positiveN

#axiomsClean ready_runs
#axiomsClean body_runs
#axiomsClean final_memory
#axiomsClean preloop_length
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Stages
