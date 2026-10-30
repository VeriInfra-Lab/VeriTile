import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IOAffine
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelMath
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.FastLayernorm.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent mathematical targets, with no source execution or memory. -/
def forwardMath (N B : Nat) (eps : ℝ) (bias : Bool) (v : ForwardValues ℝ B) :=
  forwardResult
    (fun j : Fin B => RealProof.outputSpec N (extend (forwardX v)) (extend (forwardW v))
      (fun k => if bias then extend (forwardB v) k else 0) eps j.val)
    (RealProof.invSpec N (extend (forwardX v)) eps)
    (RealProof.meanSpec N (extend (forwardX v)))
def backwardMath (N B : Nat) (v : BackwardValues ℝ B) :=
  backwardResult (fun j : Fin B => RealProof.gradientSpec N (extend (backwardX v)) (extend (backwardDY v))
    (extend (backwardW v)) (backwardMean v) (backwardInverse v) j.val)
def affineMath (rows N pid : Nat) (v : AffineValues ℝ rows) :=
  affineResult
    (fun j => if pid*128+j.val<N then
      ∑ row : Fin rows,affineDY v row j*((affineX v row j-affineMean v row)*affineInverse v row) else 0)
    (fun j => if pid*128+j.val<N then ∑ row : Fin rows,affineDY v row j else 0)

theorem forward_real (N B pid : Nat) (covered : N≤B) (eps : ℝ) (bias : Bool) :
    (forwardIO KernelMath.algebra N B pid eps bias).Implements (forwardMath N B eps bias) := by
  intro vals hd bounds hb initial hin
  obtain ⟨safe,final,ran,output,frame⟩ := forward_implements KernelMath.algebra N B pid eps bias vals hd bounds hb initial hin
  refine ⟨safe,final,ran,?_,frame⟩
  intro o j active
  rw [output o j active]
  have ho : o.val<3 := o.isLt
  rcases o with ⟨o,ho⟩
  interval_cases o
  · have formula := KernelMath.output_real N B covered eps bias
      (extend (forwardX vals)) (extend (forwardW vals)) (extend (forwardB vals)) j.1 active
    simp only [extend_fin] at formula
    have read := congrArg (Cell.read .real) formula
    simpa only [forwardGraph,forwardMath,forwardResult,Cell.read_same] using congrArg (Cell.mk .real) read
  · have formula := KernelMath.inverse_real N B covered eps (extend (forwardX vals))
    simp only [extend_fin] at formula
    change Cell.mk .real (KernelReference.cast KernelMath.algebra (KernelReference.inverse KernelMath.algebra N eps (forwardX vals))) = _
    rw [formula]
    rfl
  · have formula := KernelMath.mean_real N B covered (extend (forwardX vals))
    simp only [extend_fin] at formula
    change Cell.mk .real (KernelReference.cast KernelMath.algebra (KernelReference.mean KernelMath.algebra N (forwardX vals))) = _
    rw [formula]
    rfl

theorem backward_real (N B pid : Nat) (covered : N≤B) :
    (backwardIO KernelMath.algebra N B pid).Implements (backwardMath N B) := by
  intro vals hd bounds hb initial hin
  obtain ⟨safe,final,ran,output,frame⟩ := backward_implements KernelMath.algebra N B pid vals hd bounds hb initial hin
  refine ⟨safe,final,ran,?_,frame⟩
  intro o j active
  rw [output o j active]
  have ho : o.val<1 := o.isLt
  rcases o with ⟨o,ho⟩
  interval_cases o
  have formula := KernelMath.gradient_real N B covered
    (extend (backwardX vals)) (extend (backwardDY vals)) (extend (backwardW vals))
    (backwardMean vals) (backwardInverse vals) j.1 active
  simp only [extend_fin] at formula
  have read := congrArg (Cell.read .real) formula
  simpa only [backwardGraph,backwardMath,backwardResult,Cell.read_same] using congrArg (Cell.mk .real) read

theorem affine_real (rows N wc bc pid : Nat) (bias : Bool) :
    (affineIO KernelMath.algebra rows N wc bc pid bias).Implements (affineMath rows N pid) := by
  have eq : affineGraph KernelMath.algebra rows N pid=affineMath rows N pid := by
    funext vals o j
    have ho : o.val<2 := o.isLt
    rcases o with ⟨o,ho⟩
    interval_cases o
    all_goals
      simp only [affineGraph,affineMath,affineResult]
      rw [KernelMath.affine_real]
      simp only [KernelReference.cast,KernelMath.algebra,decide_eq_true_eq]
      split
      · rw [← Fin.sum_univ_eq_sum_range]
        simp only [extend_fin]
      · rfl
  rw [← eq]
  exact affine_implements KernelMath.algebra rows N wc bc pid bias
end
end VeriTile.Bench.Repairs.FastLayernorm.Interface
