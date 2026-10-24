import bench.repairs.tritonbench_g.rotary_emb.Proofs.Vector

namespace VeriTile.Bench.Repairs.RotaryEmb

/-- Any complete unique work order has the same in-place result. The only
writes in a pair are to that pair; coefficients and other operands stay intact. -/
theorem execute_permutation {α : Type} (s : Shape) (valid : s.Valid) (p : Ops α)
    (a b : List Work) (perm : a.Perm b) (active : ∀ w∈a,s.Active w) (unique : a.Nodup)
    (m : Memory α) : execute s p a m=execute s p b m := by
  classical
  have activeB : ∀ w∈b,s.Active w := fun w hw => active w (perm.mem_iff.mpr hw)
  have uniqueB : b.Nodup := (List.Perm.nodup_iff perm).mp unique
  funext r addr
  by_cases touch : ∃ w∈a,Writes s w r addr
  · rcases touch with ⟨w,hw,hr,ha|ha⟩
    · subst r addr
      rw [execute_readback s valid p a m active unique w hw false,
        execute_readback s valid p b m activeB uniqueB w (perm.mem_iff.mp hw) false]
    · subst r addr
      rw [execute_readback s valid p a m active unique w hw true,
        execute_readback s valid p b m activeB uniqueB w (perm.mem_iff.mp hw) true]
  · have outsideA : ∀ w∈a,¬Writes s w r addr := by simpa using touch
    have outsideB : ∀ w∈b,¬Writes s w r addr := fun w hw => outsideA w (perm.mem_iff.mpr hw)
    rw [execute_frame s p a m r addr outsideA,execute_frame s p b m r addr outsideB]

theorem any_grid_order {α : Type} (s : Shape) (valid : s.Valid) (p : Ops α)
    (order : List Work) (complete : (schedule s).Perm order) (m : Memory α) :
    execute s p order m=source s p m := by
  rw [source_eq_run s valid]
  exact (execute_permutation s valid p (schedule s) order complete (schedule_active s) (schedule_nodup s) m).symm

#axiomsClean execute_permutation
#axiomsClean any_grid_order
end VeriTile.Bench.Repairs.RotaryEmb
