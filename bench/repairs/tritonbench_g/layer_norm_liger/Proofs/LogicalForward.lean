import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Logical
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardContract
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def rowX (p : Forward.Parameters) (s : State ℝ) (i : Nat) :=
  (s.mem p.x (s.pids 0*p.xStride+i)).read .fp32
def weight (p : Forward.Parameters) (s : State ℝ) (i : Nat) := (s.mem p.weight i).read .fp32
def bias (p : Forward.Parameters) (s : State ℝ) (i : Nat) := (s.mem p.bias i).read .fp32

theorem forward_mean (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ) (hb : p.columns ≤ p.block) :
    Forward.meanValue (algebra rest) sumPrimitive p s = mean p.columns (rowX p s) := by
  unfold Forward.meanValue Forward.denominator Forward.xValues Forward.input Forward.padding
  simp only [algebra, sumPrimitive, sumValues_vector]
  change ((∑ i : Fin p.block, if i.val < p.columns then rowX p s i.val else 0) / p.columns) = _
  rw [padded_sum _ _ hb (rowX p s)]
  rfl

theorem forward_centered (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    Forward.centered (algebra rest) sumPrimitive p s i = rowX p s i.1.val-mean p.columns (rowX p s) := by
  unfold Forward.centered
  rw [forward_mean rest p s hb]
  simp [Forward.xValues, Forward.input, algebra, rowX, hi]

theorem forward_variance (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) :
    Forward.varianceValue (algebra rest) sumPrimitive p s = variance p.columns (rowX p s) := by
  unfold Forward.varianceValue
  change ((∑ i : Fin p.block, if i.val < p.columns then
    Forward.centered (algebra rest) sumPrimitive p s (i,PUnit.unit) *
    Forward.centered (algebra rest) sumPrimitive p s (i,PUnit.unit) else 0) / p.columns) = _
  have he : (fun i : Fin p.block => if i.val < p.columns then
      Forward.centered (algebra rest) sumPrimitive p s (i,PUnit.unit) *
      Forward.centered (algebra rest) sumPrimitive p s (i,PUnit.unit) else 0) =
      (fun i : Fin p.block => if i.val < p.columns then (rowX p s i.val-mean p.columns (rowX p s))^2 else 0) := by
    funext i
    by_cases hi : i.val < p.columns
    · simp [hi, forward_centered rest p s hb (i,PUnit.unit) hi, pow_two]
    · simp [hi]
  rw [he, padded_sum _ _ hb (fun i => (rowX p s i-mean p.columns (rowX p s))^2)]
  rfl

theorem forward_rstd (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) :
    Forward.rstdValue (algebra rest) sumPrimitive p s = rstd p.columns p.epsilon (rowX p s) := by
  unfold Forward.rstdValue
  rw [forward_variance rest p s hb]
  rfl

theorem forward_y (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    Forward.yValue (algebra rest) sumPrimitive p s i =
      y p.columns p.epsilon (rowX p s) (weight p s) (bias p s) i.1.val := by
  unfold Forward.yValue
  rw [forward_centered rest p s hb i hi, forward_rstd rest p s hb]
  simp [Forward.wValues, Forward.bValues, Forward.input, algebra, y, weight, bias, hi]

/-- The actual complete repaired forward execution writes the independent
population-normalization target at every logical lane. -/
theorem forward_exec (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns ≤ p.block) :
    ∃ t, FP.Structural.exec
      (VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.withSum (algebra rest) sumPrimitive)
      (Forward.kernel p) s = some t ∧
      ∀ i : TileIndex [p.block], i.1.val < p.columns →
        t.mem p.y (Forward.outputOffset p s i) =
          .mk .fp32 (y p.columns p.epsilon (rowX p s) (weight p s) (bias p s) i.1.val) := by
  obtain ⟨t,ht,hm⟩ := Forward.complete_run (algebra rest) sumPrimitive p s
  refine ⟨t,ht,?_⟩
  intro i hi
  rw [hm, Forward.reference_y (algebra rest) sumPrimitive p s i hi, forward_y rest p s hb i hi]
  rfl

#axiomsClean forward_mean
#axiomsClean forward_variance
#axiomsClean forward_rstd
#axiomsClean forward_y
#axiomsClean forward_exec
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
