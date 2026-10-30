import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealReference
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Mathematical_RealMath

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealBridge
open scoped BigOperators
noncomputable section


def padded {N B : Nat} (x : Fin N → ℝ) (j : Fin B) : ℝ :=
  if hj : j.val<N then x ⟨j.val,hj⟩ else 0

theorem sum_padded {N B : Nat} (covered : N≤B) (x : Fin N → ℝ) :
    (∑ j : Fin B,padded x j)=∑ j,x j := by
  rw [Finset.sum_fin_eq_sum_range]
  calc
    _ = ∑ i∈Finset.range B, if hi : i<N then x ⟨i,hi⟩ else 0 := by
      apply Finset.sum_congr rfl
      intro i hi
      simp [Finset.mem_range.mp hi,padded]
    _ = ∑ i∈Finset.range N, if hi : i<N then x ⟨i,hi⟩ else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono covered)
      intro i _ outside
      simp only [Finset.mem_range] at outside
      simp [outside]
    _ = _ := (Finset.sum_fin_eq_sum_range x).symm

theorem mean_eq {N B : Nat} (covered : N≤B) (x : Fin N → ℝ) :
    VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.meanSquare N (padded (B:=B) x)=(∑ j,x j^2)/(N:ℝ) := by
  have squares : (fun j : Fin B => (VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded N (padded x) j)^2)=padded (fun j => x j^2) := by
    funext j
    by_cases hj : j.val<N <;> simp [VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded,padded,hj]
  unfold VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.meanSquare
  unfold VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.squareSum
  rw [squares,sum_padded covered]

theorem inv_eq (h : Call) (x : Fin h.cols → ℝ) :
    VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.invStd h.cols h.eps (padded (B:=h.block) x)=RealReference.inv h.eps x := by
  unfold VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.invStd
  rw [mean_eq h.covered]
  rfl

def lane (h : Call) (j : Fin h.cols) : Fin h.block := ⟨j.val,j.isLt.trans_le h.covered⟩

theorem output_eq (h : Call) (x w : Fin h.cols → ℝ) (j : Fin h.cols) :
    VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.output h.gemma h.cols h.eps (padded x) (padded w) (lane h j)=RealReference.output h.gemma h.eps x w j := by
  unfold VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.output
  rw [inv_eq]
  simp [padded,lane,j.isLt,RealReference.output,RealReference.weight]

theorem gradient_eq (h : Call) (x w g : Fin h.cols → ℝ) (j : Fin h.cols) :
    VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.gradient h.gemma h.cols (RealReference.inv h.eps x)
      (padded x) (padded g) (padded w) (lane h j)=RealReference.inputGradient h.gemma h.eps x w g j := by
  let r:=RealReference.inv h.eps x
  have products : (fun k : Fin h.block =>
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded h.cols (padded g) k *
        (if h.gemma then VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded h.cols (padded w) k+1 else VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded h.cols (padded w) k) *
      (VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded h.cols (padded x) k*r))=
      padded (fun k => g k*RealReference.weight h.gemma (w k)*(x k*r)) := by
    funext k
    by_cases hk : k.val<h.cols <;>
      simp [VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded,padded,hk,RealReference.weight]
  unfold VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.gradient
  dsimp only
  change r/(h.cols:ℝ)*((h.cols:ℝ)*_ - _*(∑ k : Fin h.block,_))=_
  rw [products,sum_padded h.covered]
  simp only [VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.padded,padded,lane,j.isLt,ite_true,dite_true]
  have factored : (∑ k,g k*RealReference.weight h.gemma (w k)*(x k*r))=
      (∑ k,g k*RealReference.weight h.gemma (w k)*x k)*r := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [factored]
  have nonzero : (h.cols:ℝ)≠0 := by exact_mod_cast ne_of_gt h.positive
  unfold RealReference.inputGradient RealReference.weight
  dsimp only [r]
  field_simp [nonzero]
  <;> ring

#axiomsClean output_eq
#axiomsClean gradient_eq
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealBridge
