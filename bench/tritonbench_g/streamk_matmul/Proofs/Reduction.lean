import bench.tritonbench_g.streamk_matmul.Proofs.Publication
namespace VeriTile.Bench.Repairs.StreamK.Whole001
namespace Memory
set_option maxHeartbeats 8000000

theorem atomic_scratch_frame (add : α → α → α) (pairs : List (Fin C × Fin S))
    (s : State α A B C S E) :
    (run add (pairs.map fun ij=>.atomic ij.1 ij.2) s).scratch=s.scratch := by
  funext i
  apply run_scratch_frame
  intro e he
  obtain ⟨pair,hp,rfl⟩:=List.mem_map.mp he
  simp [WritesScratch]

/-- Exact scalar projection of an ordered atomic event sequence. It retains
the starting C value and each actual scratch load; no commutativity assumed. -/
theorem atomic_projection (add : α → α → α) (pairs : List (Fin C × Fin S))
    (s : State α A B C S E) (i : Fin C) :
    (run add (pairs.map fun ij=>.atomic ij.1 ij.2) s).output i=
      pairs.foldl (fun value pair=>if pair.1=i then add value (s.scratch pair.2) else value) (s.output i) := by
  induction pairs generalizing s with
  | nil => rfl
  | cons pair pairs ih =>
    change (run add (pairs.map fun ij=>.atomic ij.1 ij.2) (step add (.atomic pair.1 pair.2) s)).output i=_
    rw [ih]
    dsimp only [step,List.foldl_cons]
    by_cases h:pair.1=i
    · simp [h,Function.update]
    · simp only [if_neg h,Function.update_of_ne (Ne.symm h)]
end Memory
namespace Source
open Protocol Plan
set_option maxHeartbeats 12000000
noncomputable section

@[simp] theorem lanes_nodup (p : Parameters) : (lanes p).Nodup :=
  (List.nodup_finRange p.h.bm).product (List.nodup_finRange p.h.bn)

theorem fold_skip {β : Type} [DecidableEq β] (xs : List β) (x : β) (f : α → α)
    (z : α) (h : x ∉ xs) : xs.foldl (fun z y=>if y=x then f z else z) z=z := by
  induction xs with
  | nil => rfl
  | cons y ys ih =>
    have hy:y≠x:=by intro eq; apply h; simp [eq]
    have ht:x ∉ ys:=by intro hm; exact h (List.mem_cons_of_mem _ hm)
    simpa only [List.foldl_cons,if_neg hy] using ih ht

theorem fold_once {β : Type} [DecidableEq β] (xs : List β) (x : β) (f : α → α)
    (z : α) (nd : xs.Nodup) (hm : x ∈ xs) :
    xs.foldl (fun z y=>if y=x then f z else z) z=f z := by
  induction xs generalizing z with
  | nil => simp at hm
  | cons y ys ih =>
    have hn:=List.nodup_cons.mp nd
    by_cases h:y=x
    · subst y
      simp only [List.foldl_cons,if_pos rfl]
      exact fold_skip ys x f (f z) hn.1
    · have ht:x ∈ ys:=by
        rcases List.mem_cons.mp hm with hm|hm
        · exact False.elim (h hm.symm)
        · exact hm
      simp only [List.foldl_cons,if_neg h]
      exact ih z hn.2 ht

/-- A masked tile's scalar projection contains its target lane once. -/
def laneAtomic (p : Parameters) (add : α → α → α) (scratchMem : S p → α)
    (target : C p) (tile : Nat) (program : Fin p.h.programs) (value : α) (lane : Lane p) : α :=
  if h:active p tile lane then
    if output p tile lane h=target then add value (scratchMem (scratch p program lane)) else value
  else value

theorem lane_atomic_eq (p : Parameters) (add : α → α → α) (scratchMem : S p → α)
    (tile other : Nat) (ht : tile < tiles p.h) (ho : other < tiles p.h)
    (program : Fin p.h.programs) (target lane : Lane p) (ha : active p tile target) (value : α) :
    laneAtomic p add scratchMem (output p tile target ha) other program value lane=
      if other=tile ∧ lane=target then add value (scratchMem (scratch p program target)) else value := by
  unfold laneAtomic
  split
  next hactive =>
    by_cases he:output p other lane hactive=output p tile target ha
    · obtain ⟨htile,hlane⟩:=output_address_unique p other tile ho ht lane target hactive ha he
      simp only [if_pos he,htile,hlane,and_self,if_true]
    · have hne:¬(other=tile ∧ lane=target):=by
        rintro ⟨rfl,rfl⟩
        exact he rfl
      simp only [if_neg he,if_neg hne]
  next hactive =>
    have hne:¬(other=tile ∧ lane=target):=by rintro ⟨rfl,rfl⟩; exact hactive ha
    simp only [if_neg hne]

theorem lane_atomic_fold (p : Parameters) (add : α → α → α) (scratchMem : S p → α)
    (tile other : Nat) (ht : tile < tiles p.h) (ho : other < tiles p.h)
    (program : Fin p.h.programs) (target : Lane p) (ha : active p tile target) (value : α) :
    (lanes p).foldl (laneAtomic p add scratchMem (output p tile target ha) other program) value=
      if other=tile then add value (scratchMem (scratch p program target)) else value := by
  have hf:laneAtomic p add scratchMem (output p tile target ha) other program=
      fun value lane=>if other=tile ∧ lane=target then add value (scratchMem (scratch p program target)) else value := by
    funext value lane
    exact lane_atomic_eq p add scratchMem tile other ht ho program target lane ha value
  rw [hf]
  by_cases h:other=tile
  · simp only [h,true_and,if_true]
    exact fold_once (lanes p) target (fun v=>add v (scratchMem (scratch p program target))) value (lanes_nodup p) (lane_mem p target)
  · simp only [h,false_and,if_false]
    induction lanes p with
    | nil => rfl
    | cons x xs ih => simpa only [List.foldl_cons] using ih


local instance (q r I tile program : Nat) : Decidable (readsScratch q r I tile program) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

theorem fold_identity (xs : List β) (z : α) : xs.foldl (fun z _=>z) z=z := by
  induction xs with
  | nil => rfl
  | cons y ys ih => exact ih

theorem fold_congr (xs : List β) (f g : α → β → α) (z : α)
    (h : ∀ x ∈ xs, ∀ z, f z x=g z x) : xs.foldl f z=xs.foldl g z := by
  induction xs generalizing z with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [h x List.mem_cons_self z]
    exact ih _ (fun y hy z=>h y (List.mem_cons_of_mem _ hy) z)

def programFold (p : Parameters) (add : α → α → α) (scratchMem : S p → α)
    (tile : Nat) (target : Lane p) (value : α) : α :=
  (List.finRange p.h.programs).foldl (fun z program=>
    if readsScratch (select p.h).fullIterations (select p.h).partialIterations
        (select p.h).iterations tile program.val then add z (scratchMem (scratch p program target)) else z) value

theorem program_pairs_projection (p : Parameters) (add : α → α → α) (scratchMem : S p → α)
    (tile other : Nat) (ht : tile < tiles p.h) (ho : other < tiles p.h)
    (program : Fin p.h.programs) (target : Lane p) (ha : active p tile target) (value : α) :
    (programPairs p other program).foldl
      (fun z pair=>if pair.1=output p tile target ha then add z (scratchMem pair.2) else z) value=
      if other=tile then
        if readsScratch (select p.h).fullIterations (select p.h).partialIterations
            (select p.h).iterations tile program.val then add value (scratchMem (scratch p program target)) else value
      else value := by
  unfold programPairs
  split
  next hreads =>
    rw [List.foldl_flatMap]
    have hfun:(fun z lane =>
        (if h:active p other lane then [(output p other lane h,scratch p program lane)] else []).foldl
          (fun z pair=>if pair.1=output p tile target ha then add z (scratchMem pair.2) else z) z)=
        laneAtomic p add scratchMem (output p tile target ha) other program := by
      funext z lane
      dsimp only [laneAtomic]
      split <;> rfl
    rw [hfun,lane_atomic_fold p add scratchMem tile other ht ho program target ha value]
    by_cases h:other=tile
    · subst other; simp [hreads]
    · simp [h]
  next hreads =>
    simp only [List.foldl_nil]
    by_cases h:other=tile
    · subst other; simp [hreads]
    · simp [h]

theorem tile_pairs_projection (p : Parameters) (add : α → α → α) (scratchMem : S p → α)
    (tile other : Nat) (ht : tile < tiles p.h) (ho : other < tiles p.h)
    (target : Lane p) (ha : active p tile target) (value : α) :
    (tilePairs p other).foldl
      (fun z pair=>if pair.1=output p tile target ha then add z (scratchMem pair.2) else z) value=
      if other=tile then programFold p add scratchMem tile target value else value := by
  unfold tilePairs
  rw [List.foldl_flatMap]
  have hf:(fun z program => (programPairs p other program).foldl
      (fun z pair=>if pair.1=output p tile target ha then add z (scratchMem pair.2) else z) z)=
      (fun z program=>if other=tile then
        if readsScratch (select p.h).fullIterations (select p.h).partialIterations
            (select p.h).iterations tile program.val then add z (scratchMem (scratch p program target)) else z
      else z) := by
    funext z program
    exact program_pairs_projection p add scratchMem tile other ht ho program target ha z
  rw [hf]
  by_cases h:other=tile
  · simp only [h,if_true,programFold]
  · simp only [h,if_false]
    exact fold_identity _ _

/-- The generated reducer is exactly ascending-program accumulation for this
output lane. Other tile CTAs cannot affect it; no atomic commutativity or
floating-point reassociation is used. -/
theorem reduce_output_value (p : Parameters) (add : α → α → α) (initial : State α p exterior)
    (tile : Nat) (ht : tile < (select p.h).streamTiles) (target : Lane p)
    (ha : active p tile target) :
    (Memory.run add (reduceWave p) initial).output (output p tile target ha)=
      programFold p add initial.scratch tile target (initial.output (output p tile target ha)) := by
  rw [reduceWave,Memory.atomic_projection]
  unfold reducePairs
  rw [List.foldl_flatMap]
  have ht':tile < tiles p.h:=lt_of_lt_of_le ht (selected_bounded p.h)
  have hf:∀ other ∈ List.range (select p.h).streamTiles, ∀ z,
      (tilePairs p other).foldl
        (fun z pair=>if pair.1=output p tile target ha then add z (initial.scratch pair.2) else z) z=
        if other=tile then programFold p add initial.scratch tile target z else z := by
    intro other ho z
    exact tile_pairs_projection p add initial.scratch tile other ht'
      (lt_of_lt_of_le (List.mem_range.mp ho) (selected_bounded p.h)) target ha z
  have heq:(List.range (select p.h).streamTiles).foldl
      (fun z other=>(tilePairs p other).foldl
        (fun z pair=>if pair.1=output p tile target ha then add z (initial.scratch pair.2) else z) z)
      (initial.output (output p tile target ha))=
      (List.range (select p.h).streamTiles).foldl
        (fun z other=>if other=tile then programFold p add initial.scratch tile target z else z)
        (initial.output (output p tile target ha)) := by
    apply fold_congr
    intro other ho z
    exact hf other ho z
  rw [heq]
  exact fold_once _ tile (programFold p add initial.scratch tile target) _ List.nodup_range
    (List.mem_range.mpr ht)

#axiomsClean program_pairs_projection
#axiomsClean tile_pairs_projection
#axiomsClean reduce_output_value

#axiomsClean Memory.atomic_projection
#axiomsClean lane_atomic_fold
end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001
