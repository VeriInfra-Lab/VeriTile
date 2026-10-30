import bench.tritonbench_g.triton_attention.Proofs.RealTensorMath
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem probabilityAt_zero (c : Shape) (scale : ℝ) (ν : RegionName → Nat → ℝ)
    (h i j : Nat) (hz : ¬j ≤ i) : probabilityAt c scale ν h i j=0 := by
  simp [probabilityAt,backwardProbability,hz]

theorem dsAt_zero (c : Shape) (scale : ℝ) (ν : RegionName → Nat → ℝ)
    (h i j : Nat) (hz : ¬j ≤ i) : dsAt c scale ν h i j=0 := by
  simp [dsAt,backwardScoreGradient,backwardProbability,hz]

theorem tiledQ_full (c : Shape) (scale : ℝ) (ν : RegionName → Nat → ℝ) (i : Coord c) :
    tiledQ c scale ν i=∑ j ∈ Finset.range c.rows,
      dsAt c scale ν i.1.val i.2.1.val j*tensorAt c ν "K" i.1.val j i.2.2.val := by
  unfold tiledQ Shape.rows
  rw [←tiledRange_sum c.blocks]
  apply Finset.sum_congr rfl
  intro b hb
  split_ifs with h
  · rfl
  · symm
    apply Finset.sum_eq_zero
    intro j hj
    rw [dsAt_zero c scale ν _ _ _ (by omega),zero_mul]

theorem tiledK_full (c : Shape) (scale : ℝ) (ν : RegionName → Nat → ℝ) (i : Coord c) :
    tiledK c scale ν i=∑ j ∈ Finset.range c.rows,
      dsAt c scale ν i.1.val j i.2.1.val*tensorAt c ν "Q" i.1.val j i.2.2.val := by
  unfold tiledK Shape.rows
  apply upperRange_sum c.blocks (i.2.1.val/128) (Nat.le_of_lt (grid_coverage c i.2.1).1) (fun j=>dsAt c scale ν i.1.val j i.2.1.val*tensorAt c ν "Q" i.1.val j i.2.2.val)
  intro j hj
  rw [dsAt_zero c scale ν _ _ _ (by omega),zero_mul]

theorem tiledV_full (c : Shape) (scale : ℝ) (ν : RegionName → Nat → ℝ) (i : Coord c) :
    tiledV c scale ν i=∑ j ∈ Finset.range c.rows,
      probabilityAt c scale ν i.1.val j i.2.1.val*tensorAt c ν "NewDO" i.1.val j i.2.2.val := by
  unfold tiledV Shape.rows
  apply upperRange_sum c.blocks (i.2.1.val/128) (Nat.le_of_lt (grid_coverage c i.2.1).1) (fun j=>probabilityAt c scale ν i.1.val j i.2.1.val*tensorAt c ν "NewDO" i.1.val j i.2.2.val)
  intro j hj
  rw [probabilityAt_zero c scale ν _ _ _ (by omega),zero_mul]

theorem gradients_tensor (c : Shape) (scale : ℝ) (μ ν : RegionName → Nat → ℝ)
    (data : BackwardData c scale μ ν) (i : Coord c) :
    tiledQ c scale ν i=Math.gradientQ (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1)
      (headTensor c μ "V" i.1) (headTensor c μ "DO" i.1) scale i.2.1 i.2.2 ∧
    tiledK c scale ν i=Math.gradientK (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1)
      (headTensor c μ "V" i.1) (headTensor c μ "DO" i.1) scale i.2.1 i.2.2 ∧
    tiledV c scale ν i=Math.gradientV (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1)
      (headTensor c μ "DO" i.1) scale i.2.1 i.2.2 := by
  obtain ⟨hq,hk,hv⟩:=Mathematical.arbitraryShift_gradients
    (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1) (headTensor c μ "V" i.1)
    (headTensor c μ "DO" i.1) scale (headShift c ν i.1)
  rw [←hq,←hk,←hv]
  constructor
  · rw [tiledQ_full]
    rw [←Fin.sum_univ_eq_sum_range (fun j : Nat=>dsAt c scale ν i.1.val i.2.1.val j*tensorAt c ν "K" i.1.val j i.2.2.val) c.rows]
    apply Finset.sum_congr rfl
    intro j hj
    rw [ds_tensor c μ ν scale data,data.tensor_inputs c μ ν scale "K" (by simp)]
    rfl
  constructor
  · rw [tiledK_full]
    rw [←Fin.sum_univ_eq_sum_range (fun j : Nat=>dsAt c scale ν i.1.val j i.2.1.val*tensorAt c ν "Q" i.1.val j i.2.2.val) c.rows]
    apply Finset.sum_congr rfl
    intro j hj
    rw [ds_tensor c μ ν scale data,data.tensor_inputs c μ ν scale "Q" (by simp)]
    rfl
  · rw [tiledV_full]
    rw [←Fin.sum_univ_eq_sum_range (fun j : Nat=>probabilityAt c scale ν i.1.val j i.2.1.val*tensorAt c ν "NewDO" i.1.val j i.2.2.val) c.rows]
    apply Finset.sum_congr rfl
    intro j hj
    rw [probability_tensor c μ ν scale data,preprocess_tensor c μ ν scale data]

/-- Independent finite-tensor target for all four returned arrays. -/
structure TensorValues (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (final : Grid.Memory R) : Prop where
  output : ∀ i : Coord c,final "Out" (offset c i)=.mk .fp32 (some
    (Math.attention (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1) (headTensor c μ "V" i.1) scale i.2.1 i.2.2))
  dq : ∀ i : Coord c,final "DQ" (offset c i)=.mk .fp32 (some
    (Math.gradientQ (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1) (headTensor c μ "V" i.1) (headTensor c μ "DO" i.1) scale i.2.1 i.2.2))
  dk : ∀ i : Coord c,final "DK" (offset c i)=.mk .fp32 (some
    (Math.gradientK (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1) (headTensor c μ "V" i.1) (headTensor c μ "DO" i.1) scale i.2.1 i.2.2))
  dv : ∀ i : Coord c,final "DV" (offset c i)=.mk .fp32 (some
    (Math.gradientV (headTensor c μ "Q" i.1) (headTensor c μ "K" i.1) (headTensor c μ "DO" i.1) scale i.2.1 i.2.2))

theorem whole_execution_tensors (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (mem : Grid.Memory R) (μ : RegionName → Nat → ℝ)
    (inputs : ∀ r∈(["Q","K","V","DO"] : List RegionName),∀ o,(mem r o).read .fp32=some (μ r o))
    (zero : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (some 0)) :
    ∃ final,Grid.wholeRun algebra model model model c scale fw pw bw mem=some final ∧
      TensorValues c scale μ final ∧
      (∀ r o,¬Grid.Written c r o → final r o=mem r o) := by
  obtain ⟨final,saved,he,hd,hv,hframe⟩:=whole_execution_tiled c scale fw pw bw hf hp hb mem μ inputs zero
  refine ⟨final,he,?_,hframe⟩
  constructor
  · intro i
    have hout:=(hv i).1
    change final "Out" (offset c i)=.mk .fp32 (some (Mathematical.attention c.dim scale
      (tensorAt c μ "Q" i.1.val) (tensorAt c μ "K" i.1.val) (tensorAt c μ "V" i.1.val) i.2.1.val i.2.2.val)) at hout
    rw [attention_tensor] at hout
    exact hout
  · intro i
    rw [(hv i).2.1,(gradients_tensor c scale μ saved hd i).1]
  · intro i
    rw [(hv i).2.2.1,(gradients_tensor c scale μ saved hd i).2.1]
  · intro i
    rw [(hv i).2.2.2,(gradients_tensor c scale μ saved hd i).2.2]

#axiomsClean whole_execution_tensors
#specNonCircular TensorValues avoiding [FP.Structural.exec,Grid.wholeRun,Grid.wholeReference,tiledQ,tiledK,tiledV]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
