import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Spec
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference

namespace VeriTile.Bench.Repairs.MultinomialSampling.Real
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open scoped BigOperators
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 3000000

/-- Exact finite prefix sum for the source's forward sum scan. -/
def scan {shape : TileShape} (ax : Fin shape.length) (_dir : ScanDirection)
    (x : TileIndex shape → ℝ) (out : TileIndex shape) : ℝ :=
  let coord := TileShape.axisCoord shape ax out
  ∑ j ∈ Finset.range (coord.val + 1),
    if h : j < TileShape.axisDim shape ax then
      x (TileShape.replaceAxisCoord shape ax out ⟨j,h⟩) else 0

def maximum {n : Nat} (x : Fin n → ℝ) : ℝ :=
  if h : 0 < n then (Finset.univ : Finset (Fin n)).sup' ⟨⟨0,h⟩, Finset.mem_univ _⟩ x else 0

def reduceMax {shape : TileShape} (ax : Fin shape.length) :
    (keep : Bool) → (TileIndex shape → ℝ) → TileIndex (TileShape.reduceShape shape ax keep) → ℝ
  | Bool.false, x => fun out => maximum (fun k => x (TileShape.insertAxisIndex shape ax out k))
  | Bool.true, x => fun out => maximum (fun k => x (TileShape.replaceAxisIndex shape ax out k))

def primitives : Primitives ℝ :=
  { zero := fun _ _ => 0
    cast := fun _ _ _ x => x
    bits32 := fun b => ((FP.Value.mk b.bits : FP.Value .fp32).toRat?.getD 0 : ℝ)
    add := (· + ·)
    sub := (· - ·)
    lt := fun x y => decide (x < y)
    le := fun x y => decide (x ≤ y)
    scan := fun _ {_} _ ax dir x => scan ax dir x
    maximum := fun _ {_} ax keep x => reduceMax ax keep x }

theorem primitive_sample (p : Parameters) (s : State ℝ) (i : Row p) :
    sample primitives p s i = Random.randReal (seedValue p s i) (offsetValue p s i) .bits32 10 := by
  simp only [sample, primitives, Random.randReal]
  change (((Random.rand (seedValue p s i) (offsetValue p s i) .bits32 10).toRat?).getD 0 : ℝ) = _
  rw [Random.rand_toRat_randRat]
  rfl

theorem cumulative_shift (w : Nat → ℝ) (b n : Nat) :
    Spec.cumulative w b + Spec.cumulative (fun j => w (b+j)) n = Spec.cumulative w (b+n) := by
  induction n with
  | zero => simp [Spec.prefix_zero]
  | succ n ih =>
    rw [Spec.prefix_succ, ← Nat.add_assoc, Spec.prefix_succ]
    rw [← ih]
    ring

theorem cumulative_nonnegative (w : Nat → ℝ) (hw : ∀ j, 0 ≤ w j) (n : Nat) :
    0 ≤ Spec.cumulative w n := by
  exact Finset.sum_nonneg (fun j hj => hw j)

theorem cumulative_mono (w : Nat → ℝ) (hw : ∀ j, 0 ≤ w j) (a b : Nat) (hab : a ≤ b) :
    Spec.cumulative w a ≤ Spec.cumulative w b :=
  Spec.prefix_mono b w (fun j _ => hw j) a b hab le_rfl

theorem maximum_eq {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (j : Fin n)
    (h : ∀ k, x k ≤ x j) : maximum x = x j := by
  simp only [maximum, hn, ↓reduceDIte]
  apply le_antisymm
  · exact Finset.sup'_le _ _ (fun k _ => h k)
  · exact Finset.le_sup' x (Finset.mem_univ j)

private theorem fold_max {n : Nat} (v : Fin n → Int) (xs : List (Fin n)) (best : Fin n) :
    let out := xs.foldl (fun a b => if v a < v b then b else a) best
    v best ≤ v out ∧ ∀ j ∈ xs, v j ≤ v out := by
  induction xs generalizing best with
  | nil => simp
  | cons a xs ih =>
    dsimp only [List.foldl]
    let next := if v best < v a then a else best
    have h := ih next
    have hb : v best ≤ v next := by dsimp [next]; split <;> omega
    have ha : v a ≤ v next := by dsimp [next]; split <;> omega
    refine ⟨le_trans hb h.1, ?_⟩
    intro j hj
    rcases List.mem_cons.mp hj with rfl | hj
    · exact le_trans ha h.1
    · exact h.2 j hj

theorem argmax_max {n : Nat} (hn : 0 < n) (v : Fin n → Int) :
    ∃ k : Fin n, TileShape.leftmostIntArgMax v = k.val ∧ ∀ j, v j ≤ v k := by
  let out := (List.finRange n).foldl (fun a b => if v a < v b then b else a) ⟨0,hn⟩
  refine ⟨out, ?_, ?_⟩
  · simp [TileShape.leftmostIntArgMax, hn, out]
  · intro j
    exact (fold_max v (List.finRange n) ⟨0,hn⟩).2 j (List.mem_finRange j)

theorem argmax_true {n : Nat} (hn : 0 < n) (v : Fin n → Bool)
    (hex : ∃ j, v j = Bool.true) :
    ∃ k : Fin n, TileShape.leftmostIntArgMax (fun j => if v j then (1:Int) else 0) = k.val ∧ v k = Bool.true := by
  obtain ⟨k, hk, hm⟩ := argmax_max hn (fun j => if v j then (1:Int) else 0)
  refine ⟨k,hk,?_⟩
  obtain ⟨j,hj⟩ := hex
  have h := hm j
  simp only [hj, ↓reduceIte] at h
  cases e : v k <;> simp_all

theorem boolCount_positive {n : Nat} (hn : n < 2^32) (v : Fin n → Bool) :
    0 < TileShape.boolCount32 v ↔ ∃ j, v j = Bool.true := by
  have bound : (∑ j : Fin n, if v j then (1:Nat) else 0) ≤ n := by
    calc
      _ ≤ ∑ _j : Fin n, (1:Nat) := Finset.sum_le_sum (by intro j _; split <;> omega)
      _ = n := by simp
  rw [TileShape.boolCount32, Nat.mod_eq_of_lt (lt_of_le_of_lt bound hn)]
  rw [Finset.sum_pos_iff]
  simp only [Finset.mem_univ, true_and]
  constructor
  · rintro ⟨j,hj⟩
    exact ⟨j, by cases h : v j <;> simp_all⟩
  · rintro ⟨j,hj⟩
    exact ⟨j, by simp [hj]⟩

#axiomsClean primitive_sample
#axiomsClean argmax_true
#axiomsClean boolCount_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Real
