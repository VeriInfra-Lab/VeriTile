import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeWelford
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeMoments

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- The no-rounding mathematical face. Unused operations remain in rest. -/
def ideal (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    binary := fun p d op a b => match op with
      | .add => a + b
      | .sub => a - b
      | .mul => a * b
      | .div => a / b
      | _ => rest.binary p d op a b
    unary := fun p op a => match op with
      | .rsqrt | .libdeviceRsqrt => 1 / Real.sqrt a
      | _ => rest.unary p op a
    cast := fun _ _ _ x => x
    fromNat := fun _ n => n
    fromInt := fun _ n => n
    fp32Load := id }

def realEqual (a b : ℝ) : Bool := decide (a = b)

theorem ideal_update (rest : Algebra ℝ) (x : ℝ) (v : WelfordTriple ℝ) :
    reduceStep (ideal rest) Bool.false x v = realUpdate x v := by rfl

theorem ideal_combine (rest : Algebra ℝ) (a b : WelfordTriple ℝ) :
    combineTriple (ideal rest) realEqual a b = realCombine a b := by
  norm_num [combineTriple, realCombine, ideal, realEqual]

theorem ideal_tree {B : Nat} (rest : Algebra ℝ) (values : Fin B → WelfordTriple ℝ)
    (tree : ReductionTree B) :
    treeValue (ideal rest) realEqual values tree = realTree values tree := by
  induction tree with
  | leaf i => rfl
  | node l r hl hr => simp [treeValue, hl, hr, ideal_combine, realTree]

def firstSamples (s : State ℝ) (X : RegionName) (D B : Nat) : Fin B → List ℝ := fun i =>
  if i.val < D then [(s.mem X (i.val + D * s.pids 0)).read .real] else []

def advanceSamples (s : State ℝ) (X : RegionName) (D B k : Nat)
    (previous : Fin B → List ℝ) : Fin B → List ℝ := fun i =>
  if k * B + i.val < D then
    previous i ++ [(s.mem X (k * B + i.val + D * s.pids 0)).read .real]
  else previous i

def sampleRecurrence (s : State ℝ) (X : RegionName) (D B : Nat) :
    Nat → Nat → (Fin B → List ℝ) → (Fin B → List ℝ)
  | _, 0, previous => previous
  | start, count + 1, previous =>
    sampleRecurrence s X D B (start + 1) count (advanceSamples s X D B start previous)

theorem first_moments (rest : Algebra ℝ) (s : State ℝ) (X : RegionName) (D B : Nat) (i : Fin B) :
    Moments (firstLaneValues (ideal rest) s X D B i) (firstSamples s X D B i) := by
  simp only [firstLaneValues, firstSamples]
  split_ifs
  · simpa [ideal] using moments_singleton ((s.mem X (i.val + D * s.pids 0)).read .real)
  · simpa [ideal] using moments_empty

theorem advance_moments (rest : Algebra ℝ) (s : State ℝ) (X : RegionName)
    (D B k : Nat) (values : Fin B → WelfordTriple ℝ) (samples : Fin B → List ℝ)
    (h : ∀ i, Moments (values i) (samples i)) (i : Fin B) :
    Moments (advance (ideal rest) s X D B k values i) (advanceSamples s X D B k samples i) := by
  simp only [advance, advanceSamples]
  split_ifs
  · simpa [ideal_update, ideal] using
      moments_update ((s.mem X (k * B + i.val + D * s.pids 0)).read .real) (values i) (samples i) (h i)
  · exact h i

theorem recurrence_moments (rest : Algebra ℝ) (s : State ℝ) (X : RegionName)
    (D B start count : Nat) (values : Fin B → WelfordTriple ℝ) (samples : Fin B → List ℝ)
    (h : ∀ i, Moments (values i) (samples i)) :
    ∀ i, Moments (laneRecurrence (ideal rest) s X D B start count values i)
      (sampleRecurrence s X D B start count samples i) := by
  induction count generalizing start values samples with
  | zero => exact h
  | succ count ih =>
    exact ih (start + 1) _ _ (advance_moments rest s X D B start values samples h)

#axiomsClean ideal_tree
#axiomsClean recurrence_moments
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
