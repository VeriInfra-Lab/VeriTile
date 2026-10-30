import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Tree
theorem eval_congr_members {α : Type} (p : Ops α) (t : Tree) (x y : Nat → α)
    (h : ∀ i∈t.leaves,x i=y i) : t.eval p x=t.eval p y := by
  induction t with
  | zero => rfl
  | leaf i => exact h i (by simp [leaves])
  | add a b ha hb =>
    simp only [eval]
    rw [ha (fun i hi => h i (by simp [leaves,hi])),hb (fun i hi => h i (by simp [leaves,hi]))]
theorem eval_congr_legal {α : Type} (p : Ops α) (n : Nat) (t : Tree) (ht : t.Legal n)
    (x y : Nat → α) (h : ∀ i,i<n → x i=y i) : t.eval p x=t.eval p y := by
  apply eval_congr_members
  intro i hi
  apply h
  have hm:=ht.mem_iff.mp hi
  simpa using hm
#axiomsClean eval_congr_legal
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Tree
