import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Proofs.IOCorrect
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelRealTarget

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar
noncomputable section

theorem real_target (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (values : Inputs ℝ c) :
    nativeTarget (RealTarget.model reverseBias upper) c plan hasBias useSoftplus lower (RealTarget.upperWord upper) values =
      result c (RealTarget.prepared c (logicalInputs c values) hasBias useSoftplus lower upper)
        (RealTarget.cumulative c (logicalInputs c values) hasBias useSoftplus lower upper) := by
  funext o j
  fin_cases o
  · exact RealTarget.preparation reverseBias c (logicalInputs c values) hasBias useSoftplus lower upper _ _ _ _
  · have hj : j.2.2.2.1.val < c.BC := lt_of_lt_of_le j.2.2.2.1.isLt c.chunk_le_BC
    change (if h : j.2.2.2.1.val < c.BC then
      cumulativeValue (RealTarget.model reverseBias upper) c plan (logicalInputs c values)
        hasBias useSoftplus lower (RealTarget.upperWord upper)
        j.1.val j.2.1.val j.2.2.1.val ⟨j.2.2.2.1.val,h⟩ else default) =
      RealTarget.cumulative c (logicalInputs c values) hasBias useSoftplus lower upper
        j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
    rw [dif_pos hj]
    exact RealTarget.cumulative_formula reverseBias c plan (logicalInputs c values)
      hasBias useSoftplus lower upper _ _ _ ⟨j.2.2.2.1.val,hj⟩

theorem real_implements (c : Geometry) (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC)
    (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    (programIO (RealTarget.model reverseBias upper) c plan hasBias useSoftplus
      lower (RealTarget.upperWord upper)).Implements
      (fun values => result c
        (RealTarget.prepared c (logicalInputs c values) hasBias useSoftplus lower upper)
        (RealTarget.cumulative c (logicalInputs c values) hasBias useSoftplus lower upper)) := by
  intro values hd bounds hb initial hin
  obtain ⟨safe, final, ran, output, frame⟩ :=
    implements (RealTarget.model reverseBias upper) c sizes plan
      hasBias useSoftplus lower (RealTarget.upperWord upper) values hd bounds hb initial hin
  refine ⟨safe, final, ran, ?_, frame⟩
  simpa only [real_target] using output

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface
