import bench.tritonbench_g.triton_argmax.Proofs.TypedReduction

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.IntegerReduction
open VeriTile.Triton
open VeriTile.Bench.Repairs.TritonArgmax Reduction
noncomputable section

private theorem fold_spec {n : Nat} (value : Fin n → Int) (items : List (Fin n)) (initial : Fin n)
    (sorted : items.Pairwise (fun i j => i.val ≤ j.val))
    (before : ∀ i,i∈items → initial.val ≤ i.val) :
    ∀ i,i∈initial::items →
      value i ≤ value (items.foldl (fun best i => if value best<value i then i else best) initial) ∧
      (value i=value (items.foldl (fun best i => if value best<value i then i else best) initial) →
        (items.foldl (fun best i => if value best<value i then i else best) initial).val ≤ i.val) := by
  induction items generalizing initial with
  | nil =>
    intro i hi
    have same : i=initial := by simpa using hi
    subst i
    exact ⟨le_rfl,fun _ => le_rfl⟩
  | cons head tail ih =>
    obtain ⟨headBefore,tailSorted⟩ := List.pairwise_cons.mp sorted
    let next := if value initial<value head then head else initial
    have nextBefore : ∀ i,i∈tail → next.val ≤ i.val := by
      intro i hi
      dsimp only [next]
      split
      · exact headBefore i hi
      · exact before i (by simp [hi])
    have recurse := ih next tailSorted nextBefore
    have merged : ∀ i,i=initial ∨ i=head → value i ≤ value next ∧ (value i=value next → next.val ≤ i.val) := by
      intro i member
      dsimp only [next]
      split
      · rename_i better
        rcases member with rfl | rfl
        · exact ⟨better.le,fun equal => False.elim ((ne_of_lt better) equal)⟩
        · exact ⟨le_rfl,fun _ => le_rfl⟩
      · rename_i noBetter
        rcases member with rfl | rfl
        · exact ⟨le_rfl,fun _ => le_rfl⟩
        · exact ⟨le_of_not_gt noBetter,fun _ => before i (by simp)⟩
    intro i member
    change value i ≤ value (tail.foldl _ next) ∧ _
    have recurseNext := recurse next (by simp)
    rcases List.mem_cons.mp member with initialSame | member
    · have step := merged i (Or.inl initialSame)
      refine ⟨step.1.trans recurseNext.1,?_⟩
      intro same
      have left : value i=value next := le_antisymm step.1 (by rw [same]; exact recurseNext.1)
      exact (recurseNext.2 (left.symm.trans same)).trans (step.2 left)
    · rcases List.mem_cons.mp member with headSame | tailMember
      · have step := merged i (Or.inr headSame)
        refine ⟨step.1.trans recurseNext.1,?_⟩
        intro same
        have left : value i=value next := le_antisymm step.1 (by rw [same]; exact recurseNext.1)
        exact (recurseNext.2 (left.symm.trans same)).trans (step.2 left)
      · exact recurse i (by simp [tailMember])

theorem shared_correct {n : Nat} (positive : 0<n) (value : Fin n → Int) :
    ∃ bound : TileShape.leftmostIntArgMax value<n,
      IsLeastMaximum value ⟨TileShape.leftmostIntArgMax value,bound⟩ := by
  let out := (List.finRange n).foldl (fun best i => if value best<value i then i else best) ⟨0,positive⟩
  have same : TileShape.leftmostIntArgMax value=out.val := by
    simp only [TileShape.leftmostIntArgMax,dif_pos positive,out]
  have sorted : (List.finRange n).Pairwise (fun i j => i.val ≤ j.val) := by
    apply List.pairwise_iff_getElem.mpr
    intro i j hi hj less
    simpa only [List.getElem_finRange] using (Nat.le_of_lt less)
  have result := fold_spec value (List.finRange n) ⟨0,positive⟩ sorted (fun i _ => Nat.zero_le _)
  have bound : TileShape.leftmostIntArgMax value<n := by rw [same]; exact out.isLt
  have eqFin : (⟨TileShape.leftmostIntArgMax value,bound⟩ : Fin n)=out := Fin.ext same
  refine ⟨bound,?_,?_⟩
  · intro i
    rw [eqFin]
    exact (result i (by simp)).1
  · intro i equal
    change value i=value ⟨TileShape.leftmostIntArgMax value,bound⟩ at equal
    rw [eqFin] at equal
    change TileShape.leftmostIntArgMax value ≤ i.val
    rw [same]
    exact (result i (by simp)).2 equal

theorem shared_tree {n : Nat} (positive : 0<n) (value : Fin n → Int)
    (tree : Tree n) (complete : tree.Complete) :
    TileShape.leftmostIntArgMax value=(tree.pick orderedGt orderedEq value).val := by
  obtain ⟨bound,correct⟩ := shared_correct positive value
  exact congrArg Fin.val (least_maximum_unique _ _ _ correct (mathematical_maximum value tree complete))

#axiomsClean shared_correct
#axiomsClean shared_tree
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.IntegerReduction
