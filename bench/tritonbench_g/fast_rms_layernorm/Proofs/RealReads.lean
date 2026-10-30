import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealBridge

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealReads
noncomputable section
open RealBridge
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm

theorem padded_agrees (h : Call) (xs : Fin h.block → ℝ) (x : Fin h.cols → ℝ)
    (read : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,xs j=x ⟨j.val,hj⟩) :
    RealMath.padded h.cols xs=RealMath.padded h.cols (padded x) := by
  funext j
  by_cases hj : j.val<h.cols
  · simp [RealMath.padded,padded,hj,read j hj]
  · simp [RealMath.padded,padded,hj]

theorem inv_agrees (h : Call) (xs : Fin h.block → ℝ) (x : Fin h.cols → ℝ)
    (read : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,xs j=x ⟨j.val,hj⟩) :
    RealMath.invStd h.cols h.eps xs=RealReference.inv h.eps x := by
  have eq := padded_agrees h xs x read
  calc
    _ = RealMath.invStd h.cols h.eps (padded x) := by
      unfold RealMath.invStd RealMath.meanSquare RealMath.squareSum
      rw [eq]
    _ = _ := inv_eq h x

theorem output_agrees (h : Call) (xs ws : Fin h.block → ℝ) (x w : Fin h.cols → ℝ)
    (readX : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,xs j=x ⟨j.val,hj⟩)
    (readW : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,ws j=w ⟨j.val,hj⟩)
    (j : Fin h.cols) :
    RealMath.output h.gemma h.cols h.eps xs ws (lane h j)=RealReference.output h.gemma h.eps x w j := by
  unfold RealMath.output
  rw [inv_agrees h xs x readX,readX (lane h j) j.isLt,readW (lane h j) j.isLt]
  rfl

theorem gradient_agrees (h : Call) (xs ws gs : Fin h.block → ℝ) (x w g : Fin h.cols → ℝ)
    (readX : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,xs j=x ⟨j.val,hj⟩)
    (readW : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,ws j=w ⟨j.val,hj⟩)
    (readG : ∀ j : Fin h.block, ∀ hj : j.val<h.cols,gs j=g ⟨j.val,hj⟩)
    (j : Fin h.cols) :
    RealMath.gradient h.gemma h.cols (RealReference.inv h.eps x) xs gs ws (lane h j)=
      RealReference.inputGradient h.gemma h.eps x w g j := by
  have hx := padded_agrees h xs x readX
  have hw := padded_agrees h ws w readW
  have hg := padded_agrees h gs g readG
  calc
    _ = RealMath.gradient h.gemma h.cols (RealReference.inv h.eps x) (padded x) (padded g) (padded w) (lane h j) := by
      unfold RealMath.gradient
      rw [hx,hw,hg]
    _ = _ := gradient_eq h x w g j

#axiomsClean output_agrees
#axiomsClean gradient_agrees
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealReads
