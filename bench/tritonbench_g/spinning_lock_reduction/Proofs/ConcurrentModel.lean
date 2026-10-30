import bench.tritonbench_g.spinning_lock_reduction.Source

namespace VeriTile.Bench.TritonBenchG.SpinningLock
theorem cas11_returns_old_and_preserves_bits (old : I32) : cas11 old = (old, old) := by
  by_cases h : old = (1#32) <;> simp [cas11, h]

@[simp] theorem put_same {α β : Type} [DecidableEq α] (f : α → β) (k : α) (v : β) :
    put f k v k = v := by simp [put]

@[simp] theorem put_other {α β : Type} [DecidableEq α] (f : α → β) (k x : α) (v : β)
    (h : x ≠ k) : put f k v x = f x := by simp [put, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock
