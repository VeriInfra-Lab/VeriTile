import bench.tritonbench_g.rotary_transform_ops.Proofs.Vector

namespace VeriTile.Bench.Repairs.RotaryTransformOps

/-- Any complete unique work order has the same in-place result. The only
writes in a pair are to that pair; coefficients and other operands stay intact. -/
theorem execute_permutation {α : Type} (c : Config) (valid : c.Valid) (p : Ops α)
    (a b : List Work) (perm : a.Perm b) (active : ∀ w∈a,c.Active w) (unique : a.Nodup)
    (m : Memory α) : execute c p a m=execute c p b m := by
  classical
  have activeB : ∀ w∈b,c.Active w := fun w hw => active w (perm.mem_iff.mpr hw)
  have uniqueB : b.Nodup := (List.Perm.nodup_iff perm).mp unique
  funext r addr
  by_cases touch : ∃ w∈a,Writes c w r addr
  · rcases touch with ⟨w,hw,hr,ha|ha⟩
    · subst r addr
      rw [execute_readback c valid p a m active unique w hw false,
        execute_readback c valid p b m activeB uniqueB w (perm.mem_iff.mp hw) false]
    · subst r addr
      rw [execute_readback c valid p a m active unique w hw true,
        execute_readback c valid p b m activeB uniqueB w (perm.mem_iff.mp hw) true]
  · have outsideA : ∀ w∈a,¬Writes c w r addr := by simpa using touch
    have outsideB : ∀ w∈b,¬Writes c w r addr := fun w hw => outsideA w (perm.mem_iff.mpr hw)
    rw [execute_frame c p a m r addr outsideA,execute_frame c p b m r addr outsideB]

theorem any_grid_order {α : Type} (c : Config) (valid : c.Valid) (p : Ops α)
    (order : List Work) (complete : (schedule c).Perm order) (m : Memory α) :
    execute c p order m=source c p m := by
  rw [source_eq_run c valid]
  exact (execute_permutation c valid p (schedule c) order complete (schedule_active c) (schedule_nodup c) m).symm

#axiomsClean execute_permutation
#axiomsClean any_grid_order
end VeriTile.Bench.Repairs.RotaryTransformOps
