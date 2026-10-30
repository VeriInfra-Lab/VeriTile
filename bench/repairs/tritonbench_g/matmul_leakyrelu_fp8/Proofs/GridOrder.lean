import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.OutputCoverage

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Source
noncomputable section
set_option maxHeartbeats 5000000

/-- Any whole-tile serialization of the same successful one-tile-per-PID grid has
the same complete memory result. Source K order within each tile remains
fixed. The proof uses established current-read stability and concrete single
writer output coverage, not a concurrency or final-value premise. -/
theorem grid_order_independent {α β : Type} (ops : Arithmetic α β) (p : Problem) (b : Buffers)
    (layout : Layout p b) (entry : Memory α)
    (schedule : List Nat) (order : schedule.Perm (tiles p)) :
    CrossEntropy1.Schedule.run (step ops p b) schedule entry=run ops p b entry := by
  funext point
  unfold run
  rw [frozen ops p b layout entry schedule entry (by intros; rfl),
    frozen ops p b layout entry (tiles p) entry (by intros; rfl)]
  by_cases activeWrite : ∃ tile∈tiles p,writes p b tile point
  · obtain ⟨tile,present,active⟩ := activeWrite
    have witness := active
    obtain ⟨lane,ha,_⟩ := witness
    have hm : 0<p.M := by have h:=ha.1; omega
    have unique : ∀ other∈tiles p,writes p b other point → other=tile := by
      intro other ho hw
      exact tile_writes_unique p b layout hm other tile ((tiles_exact p other).mp ho)
        ((tiles_exact p tile).mp present) point hw active
    have left := CrossEntropy1.Schedule.fixed_output (writes p b) (value ops p b entry) schedule entry tile point active
      (by intro other ho hw; exact unique other (order.mem_iff.mp ho) hw)
    have right := CrossEntropy1.Schedule.fixed_output (writes p b) (value ops p b entry) (tiles p) entry tile point active unique
    rw [left,right,if_pos (order.mem_iff.mpr present),if_pos present]
  · have outside : ∀ tile∈tiles p,¬writes p b tile point := by
      intro tile ht hw
      exact activeWrite ⟨tile,ht,hw⟩
    rw [CrossEntropy1.Schedule.fixed_frame (writes p b) (value ops p b entry) schedule entry point
      (by intro tile ht; exact outside tile (order.mem_iff.mp ht)),
      CrossEntropy1.Schedule.fixed_frame (writes p b) (value ops p b entry) (tiles p) entry point outside]

#axiomsClean grid_order_independent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Source
