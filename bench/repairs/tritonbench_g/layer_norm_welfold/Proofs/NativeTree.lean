import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Accumulate
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- Ordered per-thread register fold; dynamic PTX puts each later register on
    the left, while the D=1 specialization keeps the previous partial on the left. -/
def localTree (B : Nat) (hB : 0<B) (vector8 small : Bool) (t : Nat) : ReductionTree B :=
  let count := if B=2048 then 8 else 4
  let leaf : Nat→ReductionTree B := fun i => .leaf ⟨
    (if vector8 then 8*t+i else 4*t+i%4+1024*(i/4))%B,Nat.mod_lt _ hB⟩
  (List.range (count-2)).foldl (fun prev j =>
    if small then .node prev (leaf (j+2)) else .node (leaf (j+2)) prev)
    (.node (leaf 0) (leaf 1))

def butterfly (width : Nat) (values : Nat→ReductionTree B) (delta : Nat) : Nat→ReductionTree B :=
  fun i => .node (values i) (values ((i/width)*width+((i%width) ^^^ delta)))

/-- Exact two-level PTX scalar add tree: warp butterfly, lane-0 shared stores,
    first-eight-thread shared loads, and canonical thread-0 broadcast. -/
def nativeTree (B : Nat) (hB : 0<B) (vector8 small : Bool) : ReductionTree B :=
  let warp := [16,8,4,2,1].foldl (butterfly 32) (localTree B hB vector8 small)
  let block := [4,2,1].foldl (butterfly 8) (fun w => warp (32*w))
  block 0

/-- Structurally recursive certificate sorter: unlike library mergeSort its
    kernel evaluation never depends on reducing an opaque well-founded proof. -/
def mergeFuel {α : Type} (le : α→α→Bool) : Nat→List α→List α→List α
  | 0,xs,ys => xs++ys
  | n+1,[],ys => ys
  | n+1,xs,[] => xs
  | n+1,x::xs,y::ys => if le x y then x::mergeFuel le n xs (y::ys)
      else y::mergeFuel le n (x::xs) ys

theorem mergeFuel_perm {α : Type} (le : α→α→Bool) (n : Nat) (xs ys : List α) :
    (mergeFuel le n xs ys).Perm (xs++ys) := by
  induction n generalizing xs ys with
  | zero => exact List.Perm.refl _
  | succ n ih =>
    cases xs with
    | nil => simp [mergeFuel]
    | cons x xs =>
      cases ys with
      | nil => simp [mergeFuel]
      | cons y ys =>
        simp only [mergeFuel]
        split
        · exact (ih xs (y::ys)).cons x
        · exact List.perm_cons_append_cons y (ih (x::xs) ys)

def sortFuel {α : Type} (le : α→α→Bool) : Nat→List α→List α
  | 0,xs => xs
  | n+1,xs => mergeFuel le xs.length (sortFuel le n (xs.take (xs.length/2)))
      (sortFuel le n (xs.drop (xs.length/2)))

theorem sortFuel_perm {α : Type} (le : α→α→Bool) (n : Nat) (xs : List α) :
    (sortFuel le n xs).Perm xs := by
  induction n generalizing xs with
  | zero => exact List.Perm.refl _
  | succ n ih =>
    have hp := mergeFuel_perm le xs.length
      (sortFuel le n (xs.take (xs.length/2))) (sortFuel le n (xs.drop (xs.length/2)))
    have hs := (ih (xs.take (xs.length/2))).append (ih (xs.drop (xs.length/2)))
    simpa only [sortFuel,List.take_append_drop] using hp.trans hs

theorem complete_of_sorted {B : Nat} (tree : ReductionTree B)
    (hs : sortFuel (fun a b => decide (a.val≤b.val)) 12 tree.leaves=List.finRange B) : tree.Complete := by
  have hp := sortFuel_perm (fun a b : Fin B => decide (a.val≤b.val)) 12 tree.leaves
  rw [hs] at hp
  exact hp.symm

/-- Concrete, kernel-checked leaf permutations for every observed reduction profile. -/
theorem native1024_complete : (nativeTree 1024 (by decide) Bool.false Bool.false).Complete := by
  apply complete_of_sorted
  decide +kernel

theorem native1024_one_complete : (nativeTree 1024 (by decide) Bool.false Bool.true).Complete := by
  apply complete_of_sorted
  decide +kernel

theorem native2048_complete : (nativeTree 2048 (by decide) Bool.false Bool.false).Complete := by
  apply complete_of_sorted
  decide +kernel

theorem native2048_one_complete : (nativeTree 2048 (by decide) Bool.false Bool.true).Complete := by
  apply complete_of_sorted
  decide +kernel

theorem native2048_vector_complete : (nativeTree 2048 (by decide) Bool.true Bool.false).Complete := by
  apply complete_of_sorted
  decide +kernel

#axiomsClean native1024_complete
#axiomsClean native1024_one_complete
#axiomsClean native2048_complete
#axiomsClean native2048_one_complete
#axiomsClean native2048_vector_complete
end VeriTile.Bench.Repairs.LayerNormWelfold
