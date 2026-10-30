import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealPipeline
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealGradients
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def headTensor (c : Shape) (μ : RegionName → Nat → ℝ) (r : RegionName) (h : Fin c.groups) : Math.Tensor c.rows c.dim :=
  fun i d=>tensorAt c μ r h.val i.val d.val

def headShift (c : Shape) (ν : RegionName → Nat → ℝ) (h : Fin c.groups) : Fin c.rows → ℝ :=
  fun i=>ν "M" (rowOffset c h i)

theorem score_tensor (c : Shape) (μ : RegionName → Nat → ℝ) (scale : ℝ)
    (h : Fin c.groups) (i j : Fin c.rows) :
    Math.score (headTensor c μ "Q" h) (headTensor c μ "K" h) scale i j=
      Mathematical.score c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) i.val j.val := by
  unfold Math.score Mathematical.score headTensor
  rw [Fin.sum_univ_eq_sum_range (fun d : Nat=>tensorAt c μ "Q" h.val i.val d*tensorAt c μ "K" h.val j.val d) c.dim]

theorem mass_tensor (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (h : Fin c.groups) (i : Fin c.rows) :
    Mathematical.mass i.val c.rows
      (Mathematical.score c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) i.val)
      (ν "M" (rowOffset c h i))=
    Mathematical.shiftedNormalizer (headTensor c μ "Q" h) (headTensor c μ "K" h) scale (headShift c ν h) i := by
  unfold Mathematical.mass Mathematical.shiftedNormalizer
  rw [←Fin.sum_univ_eq_sum_range (fun j : Nat=>if j ≤ i.val then
    Real.exp (Mathematical.score c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) i.val j-
      ν "M" (rowOffset c h i)) else 0) c.rows]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Math.sourceP,score_tensor,headShift]
  rfl

theorem attention_tensor (c : Shape) (μ : RegionName → Nat → ℝ) (scale : ℝ)
    (h : Fin c.groups) (i : Fin c.rows) (d : Fin c.dim) :
    Mathematical.attention c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val)
      (tensorAt c μ "V" h.val) i.val d.val=
    Math.attention (headTensor c μ "Q" h) (headTensor c μ "K" h) (headTensor c μ "V" h) scale i d := by
  rw [←Mathematical.normalized_done c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val)
    (tensorAt c μ "V" h.val) i.val d.val c.rows 0 (by omega)]
  have den : Mathematical.mass i.val c.rows
      (Mathematical.score c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) i.val) 0=
      Math.partition (headTensor c μ "Q" h) (headTensor c μ "K" h) scale i := by
    unfold Mathematical.mass Math.partition
    rw [←Fin.sum_univ_eq_sum_range (fun j : Nat=>if j ≤ i.val then
      Real.exp (Mathematical.score c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) i.val j-0) else 0) c.rows]
    apply Finset.sum_congr rfl
    intro j hj
    simp only [Math.causalWeight,score_tensor,sub_zero]
    rfl
  rw [den]
  unfold Mathematical.numerator
  rw [←Fin.sum_univ_eq_sum_range (fun j : Nat=>if j ≤ i.val then
    Real.exp (Mathematical.score c.dim scale (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) i.val j-0)*
      tensorAt c μ "V" h.val j d.val else 0) c.rows]
  simp only [Finset.sum_div,Math.attention,Math.probability]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Math.causalWeight,score_tensor,sub_zero,headTensor,Fin.le_def]
  split_ifs <;> simp [mul_div_right_comm]

theorem BackwardData.tensor_inputs (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (data : BackwardData c scale μ ν) (r : RegionName) (hr : r∈(["Q","K","V","DO"] : List RegionName)) (h : Nat) :
    tensorAt c ν r h=tensorAt c μ r h := by
  funext i d
  exact data.inputs r hr _

theorem BackwardData.normalizer (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (data : BackwardData c scale μ ν) (h : Fin c.groups) (i : Fin c.rows) :
    ν "L" (rowOffset c h i)=Mathematical.shiftedNormalizer
      (headTensor c μ "Q" h) (headTensor c μ "K" h) scale (headShift c ν h) i := by
  rw [data.denominator h i,mass_tensor]

theorem probability_tensor (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (data : BackwardData c scale μ ν) (h : Fin c.groups) (i j : Fin c.rows) :
    probabilityAt c scale ν h.val i.val j.val=
    Math.sourceP (headTensor c μ "Q" h) (headTensor c μ "K" h) scale (headShift c ν h) i j := by
  unfold probabilityAt backwardProbability
  rw [data.tensor_inputs c μ ν scale "Q" (by simp),data.tensor_inputs c μ ν scale "K" (by simp)]
  simp only [Math.sourceP,score_tensor,headShift,rowAt,rowOffset]
  rfl

theorem preprocess_tensor (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (data : BackwardData c scale μ ν) (h : Fin c.groups) (i : Fin c.rows) (d : Fin c.dim) :
    tensorAt c ν "NewDO" h.val i.val d.val=
    Math.preprocessDO (headTensor c μ "DO" h)
      (Mathematical.shiftedNormalizer (headTensor c μ "Q" h) (headTensor c μ "K" h) scale (headShift c ν h)) i d := by
  have hg:=data.gradient (h,i,d)
  rw [data.normalizer c μ ν scale h i] at hg
  exact hg

theorem delta_tensor (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (data : BackwardData c scale μ ν) (h : Fin c.groups) (i : Fin c.rows) :
    rowAt c ν "Delta" h.val i.val=
    Math.preprocessDelta
      (Math.attention (headTensor c μ "Q" h) (headTensor c μ "K" h) (headTensor c μ "V" h) scale)
      (headTensor c μ "DO" h)
      (Mathematical.shiftedNormalizer (headTensor c μ "Q" h) (headTensor c μ "K" h) scale (headShift c ν h)) i := by
  change ν "Delta" (rowOffset c h i)=_
  rw [data.delta h i,data.normalizer c μ ν scale h i]
  rw [←Fin.sum_univ_eq_sum_range (fun d : Nat=>Mathematical.attention c.dim scale
    (tensorAt c μ "Q" h.val) (tensorAt c μ "K" h.val) (tensorAt c μ "V" h.val) i.val d*
      tensorAt c μ "DO" h.val i.val d) c.dim]
  simp only [Finset.sum_div,Math.preprocessDelta,Math.preprocessDO]
  apply Finset.sum_congr rfl
  intro d hd
  rw [attention_tensor]
  exact mul_div_assoc _ _ _

theorem ds_tensor (c : Shape) (μ ν : RegionName → Nat → ℝ) (scale : ℝ)
    (data : BackwardData c scale μ ν) (h : Fin c.groups) (i j : Fin c.rows) :
    dsAt c scale ν h.val i.val j.val=
    Math.sourceDS (headTensor c μ "Q" h) (headTensor c μ "K" h) (headTensor c μ "V" h) (headTensor c μ "DO" h)
      (Math.attention (headTensor c μ "Q" h) (headTensor c μ "K" h) (headTensor c μ "V" h) scale)
      scale (Mathematical.shiftedNormalizer (headTensor c μ "Q" h) (headTensor c μ "K" h) scale (headShift c ν h))
      (headShift c ν h) i j := by
  change probabilityAt c scale ν h.val i.val j.val*((∑ d ∈ Finset.range c.dim,
    tensorAt c ν "NewDO" h.val i.val d*tensorAt c ν "V" h.val j.val d)-rowAt c ν "Delta" h.val i.val)*scale=_
  rw [probability_tensor c μ ν scale data,delta_tensor c μ ν scale data]
  unfold Math.sourceDS
  congr 3
  rw [←Fin.sum_univ_eq_sum_range (fun d : Nat=>tensorAt c ν "NewDO" h.val i.val d*tensorAt c ν "V" h.val j.val d) c.dim]
  apply Finset.sum_congr rfl
  intro d hd
  rw [preprocess_tensor c μ ν scale data,data.tensor_inputs c μ ν scale "V" (by simp)]
  rfl

theorem tiledRange_sum (n : Nat) (f : Nat → ℝ) :
    (∑ b ∈ Finset.range n,∑ j ∈ Finset.range 128,f (b*128+j))=∑ j ∈ Finset.range (n*128),f j := by
  induction n with
  | zero=>simp
  | succ n ih=>rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]

theorem upperRange_sum (n start : Nat) (hs : start≤n) (f : Nat → ℝ)
    (hz : ∀ j,j<start*128 → f j=0) :
    (∑ t ∈ Finset.range (n-start),∑ j ∈ Finset.range 128,f ((start+t)*128+j))=
      ∑ j ∈ Finset.range (n*128),f j := by
  have hh : (∑ j ∈ Finset.range (start*128),f j)=0 := Finset.sum_eq_zero (by intro j hj;exact hz j (Finset.mem_range.mp hj))
  have hsplit : n*128=start*128+(n-start)*128 := by omega
  rw [hsplit,Finset.sum_range_add,hh,zero_add]
  have ht:=tiledRange_sum (n-start) (fun j=>f (start*128+j))
  simpa only [Nat.add_mul,Nat.add_assoc] using ht

#axiomsClean attention_tensor
#axiomsClean ds_tensor
#axiomsClean upperRange_sum
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
