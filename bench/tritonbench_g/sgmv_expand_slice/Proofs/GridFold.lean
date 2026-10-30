import bench.tritonbench_g.sgmv_expand_slice.Proofs.GridGeometry
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.GridFold

def owner (shape : Shape) (segment : Segment) (r c : Nat) : Nat :=
  (r-segment.start)/32*tiles shape.columns+(c-shape.slice)/32

def step {α : Type} (shape : Shape) (segment : Segment)
    (update : Nat → Nat → α → α) (pid : Nat) (previous : Nat → Nat → α) (r c : Nat) : α :=
  if ActiveCell shape segment r c ∧ owner shape segment r c = pid then update r c (previous r c)
  else previous r c

theorem fold_once {α : Type} (shape : Shape) (segment : Segment)
    (update : Nat → Nat → α → α) (pids : List Nat) (unique : pids.Nodup)
    (previous : Nat → Nat → α) (r c : Nat) :
    (pids.foldl (fun y pid => step shape segment update pid y) previous) r c =
      if ActiveCell shape segment r c ∧ owner shape segment r c ∈ pids then update r c (previous r c)
      else previous r c := by
  induction pids generalizing previous with
  | nil => simp
  | cons pid tail ih =>
      obtain ⟨hn,ht⟩ := List.nodup_cons.mp unique
      rw [List.foldl_cons,ih ht]
      by_cases ha : ActiveCell shape segment r c
      · by_cases hp : owner shape segment r c = pid
        · have hm : owner shape segment r c ∉ tail := by simpa only [hp] using hn
          simp [step,ha,hp,hn]
        · simp [step,ha,hp]
      · simp [step,ha]

theorem owner_bound (shape : Shape) (segment : Segment) (r c : Nat)
    (h : ActiveCell shape segment r c) : owner shape segment r c < tiles segment.length*tiles shape.columns := by
  have hr : r-segment.start < segment.length := by have := h.2.1; have := h.2.2.1; omega
  have hc : c-shape.slice < shape.columns := by have := h.2.2.2.1; have := h.2.2.2.2; omega
  exact (one_dimensional_coordinate hr hc).2.2

theorem full_grid {α : Type} (shape : Shape) (segment : Segment)
    (update : Nat → Nat → α → α) (previous : Nat → Nat → α) :
    (List.range (tiles segment.length*tiles shape.columns)).foldl
      (fun y pid => step shape segment update pid y) previous =
      fun r c => if ActiveCell shape segment r c then update r c (previous r c) else previous r c := by
  funext r c
  rw [fold_once _ _ _ _ (List.nodup_range)]
  by_cases h : ActiveCell shape segment r c
  · simp [h,List.mem_range.mpr (owner_bound shape segment r c h)]
  · simp [h]

#axiomsClean fold_once
#axiomsClean owner_bound
#axiomsClean full_grid
end VeriTile.Bench.Repairs.SgmvExpandSlice.GridFold
