import bench.tritonbench_g.rms_norm_triton.Proofs.Observed

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
open Whole
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def realModel (eps : ℝ) : NativeModel ℝ where
  literal := id
  fromNat := Nat.cast
  epsilon := eps
  mul := fun x y => x*y
  add := fun x y => x+y
  fma := fun x y z => x*y+z
  div := fun x y => x/y
  sqrt := Real.sqrt
  unused := 0

def ProductValid (x : Lowered ℝ) : Prop := ∀ a b, x.product = some (a,b) → x.value = a*b

theorem multiplyInfo_real (eps : ℝ) (x y : Lowered ℝ) :
    (multiplyInfo (realModel eps) x y).value = x.value*y.value := by
  unfold multiplyInfo
  split <;> simp [realModel,mul_comm]

theorem multiplyInfo_valid (eps : ℝ) (x y : Lowered ℝ) :
    ProductValid (multiplyInfo (realModel eps) x y) := by
  intro a b equal
  have pair : (x.value,y.value) = (a,b) := Option.some.inj equal
  have hx : x.value = a := congrArg Prod.fst pair
  have hy : y.value = b := congrArg Prod.snd pair
  rw [multiplyInfo_real,hx,hy]

theorem tree_real (eps : ℝ) (xs : Fin B → Lowered ℝ)
    (products : ∀ i, ∃ x y, (xs i).product = some (x,y) ∧ (xs i).value = x*y)
    (tree : SumTree B) :
    tree.evaluate (realModel eps) xs = (tree.leaves.map (fun i => (xs i).value)).sum := by
  induction tree with
  | zero => rfl
  | leaf i => simp [SumTree.evaluate,SumTree.leaves]
  | add left right hl hr =>
    change left.evaluate (realModel eps) xs + right.evaluate (realModel eps) xs = _
    rw [hl,hr]
    simp [SumTree.leaves,List.sum_append]
  | fma i rest ih =>
    obtain ⟨x,y,hp,hv⟩ := products i
    simp only [SumTree.evaluate,hp]
    change x*y + rest.evaluate (realModel eps) xs = _
    rw [ih]
    simp only [SumTree.leaves,List.map_cons,List.sum_cons,hv]

theorem reduceInfo_real (eps : ℝ) (profile : ReductionFamily) (B lane : Nat)
    (xs : Fin B → Lowered ℝ)
    (products : ∀ i, ∃ x y, (xs i).product = some (x,y) ∧ (xs i).value = x*y) :
    (reduceInfo (realModel eps) profile B lane xs).value = ∑ i : Fin B, (xs i).value := by
  unfold reduceInfo
  split
  · rename_i h
    subst B
    simp
  · rw [tree_real eps xs products]
    rw [List.Perm.sum_eq ((profile.complete B lane).map (fun i => (xs i).value))]
    rw [← List.ofFn_eq_map,List.sum_ofFn]

theorem reduceInfo_valid (eps : ℝ) (profile : ReductionFamily) (B lane : Nat)
    (xs : Fin B → Lowered ℝ) (valid : ∀ i, ProductValid (xs i)) :
    ProductValid (reduceInfo (realModel eps) profile B lane xs) := by
  unfold reduceInfo
  split
  · exact valid _
  · intro x y h
    cases h

theorem divideInfo_real (eps : ℝ) (x y : Lowered ℝ)
    (constant : y.constant = some 1 → y.value = 1) :
    (divideInfo (realModel eps) x y).value = x.value/y.value := by
  unfold divideInfo
  split
  · rename_i h
    rw [constant h,div_one]
  · rfl

theorem divideInfo_valid (eps : ℝ) (x y : Lowered ℝ) (valid : ProductValid x) :
    ProductValid (divideInfo (realModel eps) x y) := by
  unfold divideInfo
  split
  · exact valid
  · intro a b h
    cases h

theorem addEpsilonInfo_real (eps : ℝ) (x : Lowered ℝ) (valid : ProductValid x) :
    (addEpsilonInfo (realModel eps) x).value = x.value+eps := by
  cases hp : x.product with
  | none => simp [addEpsilonInfo,hp,realModel,add_comm]
  | some pair =>
    rcases pair with ⟨a,b⟩
    simp only [addEpsilonInfo,hp,realModel,valid a b hp]

theorem reference_real (eps : ℝ) (profile : ReductionFamily) (N B : Nat) (complete : N ≤ B)
    (xs ws : Nat → ℝ) (i : Nat) :
    reference (realModel eps) profile N B xs ws i = realFormula N eps xs ws i := by
  let inputs : Fin B → Lowered ℝ := fun j =>
    ⟨if j.val < N then xs j.val else 0,none,none,Bool.false⟩
  let squares := fun j => multiplyInfo (realModel eps) (inputs j) (inputs j)
  have products : ∀ j, ∃ x y, (squares j).product = some (x,y) ∧ (squares j).value = x*y := by
    intro j
    exact ⟨(inputs j).value,(inputs j).value,rfl,multiplyInfo_real _ _ _⟩
  have valid : ∀ j, ProductValid (squares j) := fun j => multiplyInfo_valid _ _ _
  let total := reduceInfo (realModel eps) profile B i squares
  have tv : ProductValid total := reduceInfo_valid _ _ _ _ _ valid
  have totalSum : total.value = ∑ j ∈ Finset.range N, xs j*xs j := by
    rw [show total.value = ∑ j : Fin B, (squares j).value from reduceInfo_real _ _ _ _ _ products]
    have terms : (fun j : Fin B => (squares j).value) =
        (fun j : Fin B => if j.val < N then xs j.val*xs j.val else 0) := by
      funext j
      rw [multiplyInfo_real]
      dsimp [inputs]
      split_ifs <;> simp
    rw [terms]
    exact sum_prefix N B complete (fun j => xs j*xs j)
  have constant : (literalInfo (realModel eps) N).constant = some 1 →
      (literalInfo (realModel eps) N).value = 1 := by
    intro h
    exact Option.some.inj h
  have variance := divideInfo_real eps total (literalInfo (realModel eps) N) constant
  have varianceValid := divideInfo_valid eps total (literalInfo (realModel eps) N) tv
  have radicand := addEpsilonInfo_real eps (divideInfo (realModel eps) total (literalInfo (realModel eps) N)) varianceValid
  change (multiplyInfo (realModel eps)
    (multiplyInfo (realModel eps) ⟨xs i,none,none,Bool.false⟩
      (divideInfo (realModel eps) (literalInfo (realModel eps) 1)
        ⟨Real.sqrt (addEpsilonInfo (realModel eps) (divideInfo (realModel eps) total (literalInfo (realModel eps) N))).value,
          none,none,Bool.false⟩)) ⟨ws i,none,none,Bool.false⟩).value = _
  rw [multiplyInfo_real,multiplyInfo_real,divideInfo_real eps _ _ (by intro h; cases h),radicand,variance,totalSum]
  rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
