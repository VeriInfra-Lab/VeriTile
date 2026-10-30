import bench.tritonbench_g.attention_score.Proofs.Reused.NativePhysical
import bench.tritonbench_g.attention_score.Proofs.Reused.LogicalFormula

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeIdeal
open VeriTile Triton NativeInstructions NativeStaging NativeReduction
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Exact-real meaning for individual ISA ports. The eventual numerical
obligation relates each actual fixed instruction to these scalar operations. -/
def isa (decode : BitVec 32 → ℝ) : ISA ℝ where
  bits := decode
  literal32 := id
  addRN := (·+·)
  mulRN := (·*·)
  negateSign := Neg.neg
  ex2ApproxFTZ := pow2
  fmaRN := fun a b c => a*b+c
  mmaTF32 := fun _ _ a b c => c+∑ i : Fin 8, a i*b i

@[simp] theorem eval_bits (decode : BitVec 32 → ℝ) (b : BitVec 32) :
    (Graph.bits b).eval (isa decode)=decode b := rfl
@[simp] theorem eval_literal (decode : BitVec 32 → ℝ) (x : ℝ) :
    (Graph.literal32 x).eval (isa decode)=x := rfl
@[simp] theorem eval_add (decode : BitVec 32 → ℝ) (a b : Graph) :
    (Graph.addRN a b).eval (isa decode)=a.eval (isa decode)+b.eval (isa decode) := rfl
@[simp] theorem eval_mul (decode : BitVec 32 → ℝ) (a b : Graph) :
    (Graph.mulRN a b).eval (isa decode)=a.eval (isa decode)*b.eval (isa decode) := rfl
@[simp] theorem eval_neg (decode : BitVec 32 → ℝ) (a : Graph) :
    (Graph.negateSign a).eval (isa decode)= -a.eval (isa decode) := rfl
@[simp] theorem eval_exp (decode : BitVec 32 → ℝ) (a : Graph) :
    (Graph.ex2ApproxFTZ a).eval (isa decode)=pow2 (a.eval (isa decode)) := rfl
@[simp] theorem eval_fma (decode : BitVec 32 → ℝ) (a b c : Graph) :
    (Graph.fmaRN a b c).eval (isa decode)=a.eval (isa decode)*b.eval (isa decode)+c.eval (isa decode) := rfl

@[simp] theorem eval_ite (decode : BitVec 32 → ℝ) (P : Prop) [Decidable P] (a b : Graph) :
    (if P then a else b).eval (isa decode)=(if P then a.eval (isa decode) else b.eval (isa decode)) := by
  by_cases h : P <;> simp [h]

theorem list_range_sum (f : Nat → ℝ) (n : Nat) :
    ((List.range n).map f).sum=∑ i∈Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih]

def feature (decode : BitVec 32 → ℝ) (D : Nat) (q k : Fin D → Graph) (d : Nat) : ℝ :=
  if h : d<D then (q ⟨d,h⟩).eval (isa decode)*(k ⟨d,h⟩).eval (isa decode) else 0

theorem step_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (D row col block : Nat)
    (q k : Fin D → Graph) (seed : Graph) :
    (dotStep D row col block q k seed).eval (isa decode)=seed.eval (isa decode)+
      ∑ lane : Fin 8, feature decode D q k (block*8+lane.val) := by
  simp only [dotStep,Graph.eval,isa]
  congr 1
  apply Finset.sum_congr rfl
  intro lane _
  by_cases h : block*8+lane.val<D
  · simp [h,feature]
    rfl
  · simp [h,feature,NativeInstructions.zero,Graph.eval,isa,hz]
    exact hz

theorem fold_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (D row col : Nat)
    (q k : Fin D → Graph) (blocks : List Nat) (seed : Graph) :
    (blocks.foldl (fun seed block => dotStep D row col block q k seed) seed).eval (isa decode)=
      seed.eval (isa decode)+(blocks.map (fun b => ∑ i : Fin 8, feature decode D q k (b*8+i.val))).sum := by
  induction blocks generalizing seed with
  | nil => simp
  | cons b rest ih =>
      simp only [List.foldl_cons,ih,step_eval decode hz,List.map_cons,List.sum_cons]
      ring

/-- Exact-real interpretation of the explicit fixed-eight-operand MMA chain
is the full D-feature sum. The proof covers arbitrary D, including its tail. -/
theorem dot_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (D row col : Nat)
    (q k : Fin D → Graph) (seed : Graph) :
    (dot D row col q k seed).eval (isa decode)=seed.eval (isa decode)+
      ∑ d : Fin D, (q d).eval (isa decode)*(k d).eval (isa decode) := by
  rw [dot,fold_eval decode hz,list_range_sum]
  congr 1
  let f := feature decode D q k
  have hp := LogicalFormula.partition f ((D+7)/8) 8
  rw [Fin.sum_univ_eq_sum_range (fun b => ∑ r : Fin 8, f (b*8+r.val)) ((D+7)/8)] at hp
  rw [hp]
  have hcover : D≤(D+7)/8*8 := by omega
  have hcut : (∑ d∈Finset.range ((D+7)/8*8), f d)=∑ d∈Finset.range D, f d := by
    symm
    apply Finset.sum_subset (Finset.range_mono hcover)
    intro d _ hd
    simp only [Finset.mem_range,not_lt] at hd
    simp [f,feature,Nat.not_lt.mpr hd]
  rw [hcut,←Fin.sum_univ_eq_sum_range f D]
  apply Finset.sum_congr rfl
  intro d _
  simp [f,feature,d.isLt]

theorem tree_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (xs : Nat → Graph) (tree : Tree) :
    (tree.value NativeInstructions.zero xs Graph.move Graph.addRN).eval (isa decode)=
      (tree.leaves.map (fun i => (xs i).eval (isa decode))).sum := by
  induction tree with
  | zero => simpa only [Tree.value,Tree.leaves,NativeInstructions.zero,Graph.eval,isa,List.map_nil,List.sum_nil] using hz
  | input i => simp [Tree.value,Tree.leaves]
  | move route t ih => exact ih
  | add a b ha hb =>
      change (a.value NativeInstructions.zero xs Graph.move Graph.addRN).eval (isa decode)+
        (b.value NativeInstructions.zero xs Graph.move Graph.addRN).eval (isa decode)=_
      rw [ha,hb]
      simp [Tree.leaves]

theorem reduction_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (profile : Fin 3) (xs : Nat → Graph) :
    ((plan profile).value NativeInstructions.zero xs Graph.move Graph.addRN).eval (isa decode)=
      ∑ i : Fin (width profile), (xs i.val).eval (isa decode) := by
  rw [tree_eval decode hz]
  have hp := List.Perm.sum_eq ((coverage profile).map (fun i => (xs i).eval (isa decode)))
  rw [hp,list_range_sum,←Fin.sum_univ_eq_sum_range]

#axiomsClean dot_eval
#axiomsClean reduction_eval
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeIdeal
