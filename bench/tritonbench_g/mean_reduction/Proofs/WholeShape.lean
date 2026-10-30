import bench.tritonbench_g.mean_reduction.Proofs.WholeMath

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
noncomputable section

/-- The source shape-mutation loop, with its scalar product factored in the
same left-to-right order. No repeated normalized axes are permitted. -/
def shapeLoop {rank : Nat} (shape : Fin rank → Nat) : List (Fin rank) →
    Nat × (Fin rank → Nat)
  | [] => (1,shape)
  | i::is => let tail := shapeLoop (Function.update shape i 1) is
             (shape i * tail.1, tail.2)

theorem shapeLoop_values {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (j : Fin rank) :
    (shapeLoop shape dims).2 j = if j ∈ dims then 1 else shape j := by
  induction dims generalizing shape with
  | nil => simp [shapeLoop]
  | cons i is ih =>
    simp only [shapeLoop,ih,List.mem_cons]
    by_cases h : j = i
    · subst j; simp
    · simp [h,Function.update_of_ne h]

theorem shapeLoop_columns {rank : Nat} (shape : Fin rank → Nat)
    (dims : List (Fin rank)) (nodup : dims.Nodup) :
    (shapeLoop shape dims).1 = columns shape dims := by
  induction dims generalizing shape with
  | nil => rfl
  | cons i is ih =>
    simp only [shapeLoop,ih _ nodup.tail,columns,List.map_cons,List.prod_cons]
    congr 1
    congr 1
    apply List.map_congr_left
    intro j hj
    have hji : j ≠ i := by
      intro h
      subst j
      exact (List.nodup_cons.mp nodup).1 hj
    simp [Function.update_of_ne hji]

theorem product_partition {α : Type} (axes : List α) (shape : α → Nat)
    (p : α → Prop) [DecidablePred p] :
    (axes.map shape).prod =
      ((axes.filter fun i => decide (¬p i)).map shape).prod *
      ((axes.filter fun i => decide (p i)).map shape).prod := by
  induction axes with
  | nil => simp
  | cons i is ih =>
    by_cases h : p i <;> simp [h,ih,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

theorem selectedAxes_perm {rank : Nat} (dims : List (Fin rank)) (nodup : dims.Nodup) :
    ((List.finRange rank).filter (fun i => decide (i ∈ dims))).Perm dims := by
  apply (List.perm_ext_iff_of_nodup ((List.nodup_finRange rank).filter _) nodup).mpr
  intro i
  simp

def originalNumel {rank : Nat} (shape : Fin rank → Nat) : Nat :=
  ((List.finRange rank).map shape).prod

theorem source_numel {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (nodup : dims.Nodup) :
    originalNumel shape = rows shape dims * columns shape dims := by
  unfold originalNumel
  rw [product_partition (List.finRange rank) shape (fun i => i ∈ dims)]
  rw [((selectedAxes_perm dims nodup).map shape).prod_eq]
  rfl

theorem host_M {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (nodup : dims.Nodup) (positive : 0 < columns shape dims) :
    originalNumel shape / columns shape dims = rows shape dims := by
  rw [source_numel shape dims nodup]
  exact Nat.mul_div_cancel _ positive

def outputShape {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (keepdim : Bool) : List Nat :=
  if keepdim then (List.finRange rank).map (fun i => if i ∈ dims then 1 else shape i)
  else (batchAxes dims).map shape

theorem product_kept {α : Type} (axes : List α) (shape : α → Nat)
    (p : α → Prop) [DecidablePred p] :
    (axes.map (fun i => if p i then 1 else shape i)).prod =
      ((axes.filter (fun i => decide (¬p i))).map shape).prod := by
  induction axes with
  | nil => simp
  | cons i is ih => by_cases h : p i <;> simp [h,ih]

theorem output_numel {rank : Nat} (shape : Fin rank → Nat) (dims : List (Fin rank))
    (keepdim : Bool) : (outputShape shape dims keepdim).prod = rows shape dims := by
  cases keepdim
  · rfl
  · exact product_kept _ _ _

/-- Python's modulo normalization, including negative and oversized signed axes. -/
def normalizeAxis (rank : Nat) (positive : 0 < rank) (d : Int) : Fin rank :=
  ⟨(d % (rank : Int)).toNat, by
    have hn : 0 < (rank : Int) := by exact_mod_cast positive
    have h0 := Int.emod_nonneg d (Int.ne_of_gt hn)
    have h1 := Int.emod_lt_of_pos d hn
    omega⟩

theorem normalizeAxis_nonnegative (rank : Nat) (positive : 0 < rank)
    (i : Fin rank) : normalizeAxis rank positive (i.val : Int) = i := by
  apply Fin.ext
  simp [normalizeAxis,Int.emod_eq_of_lt (Int.natCast_nonneg _) (Int.ofNat_lt.mpr i.isLt)]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
