import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Real
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware

namespace VeriTile.Bench.Repairs.MultinomialSampling.NumericTrees
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

def decode (x : Word) : ℝ := ((f32 x).toRat?.getD 0 : ℝ)
def Finite (x : Word) : Prop := (f32 x).toRat?.isSome = Bool.true

def evalReal (op : ℝ → ℝ → ℝ) (x : Fin n → ℝ) : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree n → ℝ
  | .leaf i => x i
  | .join a b => op (evalReal op x a) (evalReal op x b)

/-- A recursive collection of local binary-node conditions. Each conjunct
names the actual two word operands of that node. No whole scan equation is
assumed; the tree theorem below derives its value from these local conditions. -/
def Local (op : Word → Word → Word) (realOp : ℝ → ℝ → ℝ) (x : Fin n → Word) : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree n → Prop
  | .leaf i => Finite (x i)
  | .join a b => Local op realOp x a ∧ Local op realOp x b ∧
      Finite (op (a.eval op x) (b.eval op x)) ∧
      decode (op (a.eval op x) (b.eval op x)) = realOp (decode (a.eval op x)) (decode (b.eval op x))

theorem eval_decode (op : Word → Word → Word) (realOp : ℝ → ℝ → ℝ)
    (x : Fin n → Word) (t : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree n) (h : Local op realOp x t) :
    decode (t.eval op x) = evalReal realOp (fun i => decode (x i)) t := by
  induction t with
  | leaf i => rfl
  | join a b ia ib =>
    rw [Tree.eval,h.2.2.2,ia h.1,ib h.2.1]
    rfl

theorem sum_leaves (x : Fin n → ℝ) (t : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree n) :
    evalReal (·+·) x t = (t.leaves.map x).sum := by
  induction t with
  | leaf i => simp [evalReal,Tree.leaves]
  | join a b ia ib => simp [evalReal,Tree.leaves,ia,ib]

theorem sum_plan (op : Word → Word → Word) (x : Fin n → Word) (t : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree n)
    (permutation : t.leaves.Perm (List.finRange n)) (h : Local op (·+·) x t) :
    decode (t.eval op x) = ∑ i : Fin n, decode (x i) := by
  rw [eval_decode op (·+·) x t h,sum_leaves]
  rw [(permutation.map (fun i => decode (x i))).sum_eq]
  rw [← List.ofFn_eq_map]
  exact List.sum_ofFn

theorem max_member (x : Fin n → ℝ) (t : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree n) :
    (∃ i ∈ t.leaves, evalReal max x t = x i) ∧
    (∀ i ∈ t.leaves, x i ≤ evalReal max x t) := by
  induction t with
  | leaf i => simp [evalReal,Tree.leaves]
  | join a b ia ib =>
    rcases ia with ⟨⟨i,hi,he⟩,ha⟩
    rcases ib with ⟨⟨j,hj,hf⟩,hb⟩
    constructor
    · by_cases h : x i ≤ x j
      · refine ⟨j,by simp [Tree.leaves,hj],?_⟩
        simp [evalReal,he,hf,max_eq_right h]
      · refine ⟨i,by simp [Tree.leaves,hi],?_⟩
        simp [evalReal,he,hf,max_eq_left (le_of_not_ge h)]
    · intro k hk
      rcases List.mem_append.mp hk with hk | hk
      · exact le_trans (ha k hk) (le_max_left _ _)
      · exact le_trans (hb k hk) (le_max_right _ _)

theorem max_plan (op : Word → Word → Word) (x : Fin (n+1) → Word) (t : VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Tree (n+1))
    (permutation : t.leaves.Perm (List.finRange (n+1))) (h : Local op max x t) :
    decode (t.eval op x) = Real.maximum (fun i => decode (x i)) := by
  rw [eval_decode op max x t h]
  obtain ⟨⟨j,hj,he⟩,hm⟩ := max_member (fun i => decode (x i)) t
  rw [he]
  symm
  apply Real.maximum_eq (Nat.succ_pos n) _ j
  intro i
  rw [← he]
  exact hm i (permutation.mem_iff.mpr (List.mem_finRange i))

def ScanLocal (plans : Plans) (p : Parameters) (x : Lane p → Word) (i : Lane p) : Prop :=
  Local (scanOp .sum) (·+·)
    (fun j : Fin (i.2.1.val+1) => x (i.1,⟨j.val,by have := j.isLt; have := i.2.1.isLt; omega⟩,PUnit.unit))
    (plans.scanTree i.2.1.val)

theorem scan_decode (plans : Plans) (p : Parameters) (x : Lane p → Word) (i : Lane p)
    (h : ScanLocal plans p x i) :
    decode (forwardScan plans .sum ⟨1,by simp⟩ x i) =
      Real.scan ⟨1,by simp⟩ .forward (fun j => decode (x j)) i := by
  change decode ((plans.scanTree i.2.1.val).eval (scanOp .sum)
    (fun j : Fin (i.2.1.val+1) => x (i.1,⟨j.val,by have := j.isLt; have := i.2.1.isLt; omega⟩,PUnit.unit))) = _
  rw [sum_plan _ _ _ (plans.scanValid _) h]
  let f : Nat → ℝ := fun k => if hk : k < p.blockTokens then decode (x (i.1,⟨k,hk⟩,PUnit.unit)) else 0
  change _ = ∑ k ∈ Finset.range (i.2.1.val+1), f k
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro j _
  have hj : j.val < p.blockTokens := by have := j.isLt; have := i.2.1.isLt; omega
  simp [f,hj]

def MaxLocal (plans : Plans) : {n : Nat} → (Fin n → Word) → Prop
  | 0,_ => True
  | n+1,x => Local maxnum max x (plans.maxTree n)

theorem maximum_decode (plans : Plans) {n : Nat} (hn : 0 < n) (x : Fin n → Word)
    (h : MaxLocal plans x) : decode (maximum plans x) = Real.maximum (fun i => decode (x i)) := by
  cases n with
  | zero => omega
  | succ n => exact max_plan maxnum x (plans.maxTree n) (plans.maxValid n) h

#axiomsClean eval_decode
#axiomsClean sum_plan
#axiomsClean max_plan
#axiomsClean scan_decode
#axiomsClean maximum_decode
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.NumericTrees
