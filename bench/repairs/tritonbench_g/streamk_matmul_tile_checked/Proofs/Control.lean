import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Decode
import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Protocol

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

theorem source_loop_unfold (p : Source.Parameters) (program stop current : Nat) :
    chunks sourcePlan.first (put (parameters p) "pid" program) stop current =
    if current<stop then
      if current<Plan.next (Plan.select p.h).iterations stop current ∧
          Plan.next (Plan.select p.h).iterations stop current≤stop then do
        let rest ← chunks sourcePlan.first (put (parameters p) "pid" program) stop
          (Plan.next (Plan.select p.h).iterations stop current)
        some (⟨current,Plan.next (Plan.select p.h).iterations stop current⟩::rest)
      else none
    else some [] := by
  rw [chunks]
  rfl

theorem source_chunks (p : Source.Parameters) (program stop current : Nat)
    (hI : 0<(Plan.select p.h).iterations) :
    chunks sourcePlan.first (put (parameters p) "pid" program) stop current =
      some (Protocol.chunks (Plan.select p.h).iterations stop current hI) := by
  rw [source_loop_unfold]
  by_cases h : current<stop
  · have progress := Plan.next_progress (Plan.select p.h).iterations stop current hI h
    rw [if_pos h,if_pos progress,
      source_chunks p program stop (Plan.next (Plan.select p.h).iterations stop current) hI,
      Protocol.chunks_cons _ _ _ hI h]
    rfl
  · rw [if_neg h,Protocol.chunks_nil _ _ _ hI (by omega)]
termination_by stop-current
decreasing_by
  have progress := Plan.next_progress (Plan.select p.h).iterations stop current hI h
  exact Nat.sub_lt_sub_left h progress.1

theorem first_pid (p : Source.Parameters) (program : Nat) :
    sourcePlan.first.pid.eval (parameters p) ⟨program,0,0,0⟩=program := by rfl

theorem first_start (p : Source.Parameters) (program : Nat) :
    sourcePlan.first.start.eval (put (parameters p) "pid" program) ⟨program,0,0,0⟩=
      Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations program := by rfl

theorem first_stop (p : Source.Parameters) (program : Nat) :
    sourcePlan.first.stop.eval (put (parameters p) "pid" program) ⟨program,0,0,0⟩=
      Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations (program+1) := by rfl

#axiomsClean source_loop_unfold
#axiomsClean source_chunks
#axiomsClean first_pid
#axiomsClean first_start
#axiomsClean first_stop
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
