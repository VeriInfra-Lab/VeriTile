import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_FPWelfordTree

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FPStages
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- Raw moment invariant, independent of memory and program execution. -/
structure Moments (v : WelfordTriple ℝ) (xs : List ℝ) : Prop where
  weight : v.weight = xs.length
  sum : v.mean * v.weight = xs.sum
  squares : v.m2 + v.mean ^ 2 * v.weight = (xs.map (fun x => x ^ 2)).sum
  empty : xs = [] → v.mean = 0

def realCombine (l r : WelfordTriple ℝ) : WelfordTriple ℝ :=
  let delta := if l.mean = r.mean then 0 else r.mean - l.mean
  let weight := l.weight + r.weight
  let ratio := if weight = 0 then 0 else r.weight / weight
  ⟨l.mean + delta * ratio,
    (l.m2 + r.m2) + ((delta * delta) * l.weight) * ratio, weight⟩

theorem delta_real (a b : ℝ) : (if a = b then 0 else b - a) = b - a := by
  split_ifs with h <;> simp_all

theorem moments_empty : Moments (⟨0,0,0⟩ : WelfordTriple ℝ) [] := by
  constructor <;> simp

theorem moments_singleton (x : ℝ) : Moments (⟨x,0,1⟩ : WelfordTriple ℝ) [x] := by
  constructor <;> simp

theorem moments_combine (l r : WelfordTriple ℝ) (xs ys : List ℝ)
    (hl : Moments l xs) (hr : Moments r ys) :
    Moments (realCombine l r) (xs ++ ys) := by
  have wl : 0 ≤ l.weight := by rw [hl.weight]; positivity
  have wr : 0 ≤ r.weight := by rw [hr.weight]; positivity
  by_cases hz : l.weight + r.weight = 0
  · have hwl : l.weight = 0 := by linarith
    have hwr : r.weight = 0 := by linarith
    have hx : xs = [] := by simpa [hl.weight] using hwl
    have hy : ys = [] := by simpa [hr.weight] using hwr
    have hm1 := hl.empty hx
    have hm2 := hr.empty hy
    have hs1 : l.m2 = 0 := by simpa [hx, hwl] using hl.squares
    have hs2 : r.m2 = 0 := by simpa [hy, hwr] using hr.squares
    simpa [realCombine, hx, hy, hm1, hm2, hs1, hs2, hwl, hwr] using moments_empty
  · have hc : (xs ++ ys) ≠ [] := by
      intro he
      have hlen : (xs ++ ys).length = 0 := by rw [he]; rfl
      have hxlen : xs.length = 0 := by simp only [List.length_append] at hlen; omega
      have hylen : ys.length = 0 := by simp only [List.length_append] at hlen; omega
      apply hz
      simp [hl.weight, hr.weight, hxlen, hylen]
    constructor
    · simp [realCombine, hl.weight, hr.weight]
    · simp only [realCombine, delta_real, if_neg hz, List.sum_append]
      rw [← hl.sum, ← hr.sum]
      field_simp [hz]
      <;> ring
    · simp only [realCombine, delta_real, if_neg hz, List.map_append, List.sum_append]
      rw [← hl.squares, ← hr.squares]
      field_simp [hz]
      <;> ring
    · intro he
      exact (hc he).elim

/-- Welford's independent closed-form mean and centered square sum. -/
theorem moments_result (v : WelfordTriple ℝ) (xs : List ℝ)
    (h : Moments v xs) (hne : xs ≠ []) :
    v.mean = xs.sum / xs.length ∧
      v.m2 = (xs.map (fun x => (x - xs.sum / xs.length) ^ 2)).sum := by
  have hw : v.weight ≠ 0 := by
    rw [h.weight]
    have hp : 0 < xs.length := by cases xs <;> simp_all
    exact_mod_cast ne_of_gt hp
  have hm : v.mean = xs.sum / xs.length := by
    rw [← h.weight]
    exact (eq_div_iff hw).mpr h.sum
  refine ⟨hm, ?_⟩
  have expansion (a : ℝ) (ys : List ℝ) :
      (ys.map (fun x => (x - a) ^ 2)).sum =
        (ys.map (fun x => x ^ 2)).sum - 2 * a * ys.sum + ys.length * a ^ 2 := by
    induction ys with
    | nil => simp
    | cons x ys ih => simp [ih]; ring
  rw [expansion, ← hm, ← h.sum, ← h.squares, ← h.weight]
  ring

def realUpdate (x : ℝ) (v : WelfordTriple ℝ) : WelfordTriple ℝ :=
  let delta := x - v.mean
  let weight := v.weight + 1
  let mean := v.mean + delta / weight
  ⟨mean, v.m2 + delta * (x - mean), weight⟩

theorem real_update_combine (x : ℝ) (v : WelfordTriple ℝ) (hw : 0 ≤ v.weight) :
    realUpdate x v = realCombine v ⟨x,0,1⟩ := by
  have hne : v.weight + 1 ≠ 0 := by linarith
  simp only [realUpdate, realCombine, delta_real, if_neg hne]
  congr 1 <;> field_simp <;> ring

theorem moments_update (x : ℝ) (v : WelfordTriple ℝ) (xs : List ℝ)
    (hv : Moments v xs) : Moments (realUpdate x v) (xs ++ [x]) := by
  have hw : 0 ≤ v.weight := by rw [hv.weight]; positivity
  rw [real_update_combine x v hw]
  exact moments_combine v ⟨x,0,1⟩ xs [x] hv (moments_singleton x)

def realTree {B : Nat} (values : Fin B → WelfordTriple ℝ) : ReductionTree B → WelfordTriple ℝ
  | .leaf i => values i
  | .node l r => realCombine (realTree values l) (realTree values r)

def treeSamples {B : Nat} (lanes : Fin B → List ℝ) : ReductionTree B → List ℝ
  | .leaf i => lanes i
  | .node l r => treeSamples lanes l ++ treeSamples lanes r

theorem moments_tree {B : Nat} (values : Fin B → WelfordTriple ℝ)
    (lanes : Fin B → List ℝ) (h : ∀ i, Moments (values i) (lanes i))
    (tree : ReductionTree B) : Moments (realTree values tree) (treeSamples lanes tree) := by
  induction tree with
  | leaf i => exact h i
  | node l r hl hr => exact moments_combine _ _ _ _ hl hr

theorem tree_samples_flatMap {B : Nat} (lanes : Fin B → List ℝ) (tree : ReductionTree B) :
    treeSamples lanes tree = tree.leaves.flatMap lanes := by
  induction tree with
  | leaf i => simp [treeSamples, ReductionTree.leaves]
  | node l r hl hr => simp [treeSamples, ReductionTree.leaves, hl, hr]

#axiomsClean moments_update
#axiomsClean moments_tree
#axiomsClean moments_combine
#axiomsClean moments_result
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
