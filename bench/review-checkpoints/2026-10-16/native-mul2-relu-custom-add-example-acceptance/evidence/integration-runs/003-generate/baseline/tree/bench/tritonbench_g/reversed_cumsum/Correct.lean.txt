import bench.tritonbench_g.reversed_cumsum.Kernels
import bench.tritonbench_g.reversed_cumsum.Proofs.RealSurface
import bench.tritonbench_g.reversed_cumsum.Proofs.RealIO
import VeriTile.Meta.Specification

/-! Successful full-source real execution and independent initial-memory
mathematics, standard tagged realization, and finite allocation IO/frame.
The general region result includes the source's unspecified time-tail values;
the finite bridge explicitly specializes to a zero oracle. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.Correctness
open VeriTile VeriTile.Triton RealProgram RealLoop
noncomputable section

def kernel (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) : ComputeKernel :=
  (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType

/-- A complete per-program contract. The output is defined from initial input
and padding values, never from a prepared intermediate tile or a final readback.
The finite clause preserves every physical cell outside active writes. -/
def Implements (k : ComputeKernel) (p : Layout) (s o : RegionName) : Prop :=
  (∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
    ∃ final, exec k.toAlgKernel initial = some final ∧ RealExecution.Post p s o initial final) ∧
  (∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
    ComputeCorrect.RealizesFrame_without_Rounding k initial
      (RealSurface.writes p (RealInit.launch initial) o) (RealSurface.expectedCell p s initial)) ∧
  (∀ (A : FlatAlloc), A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
    ∀ initial : BlockState, s ≠ o → 0 < p.BT → NoCollision p (RealInit.launch initial) →
      RealSafety.Bounds p (RealInit.launch initial) s o A.extent →
      initial.undef = (fun _ _ => 0) →
      ∃ final, exec (A.flattenKernel k.toAlgKernel) (A.flattenState initial) = some final ∧
        RealIO.Post A p s o initial final)

theorem source_implements (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    Implements (kernel swapped dtype out_dtype p s o) p s o := by
  refine ⟨?_,?_,?_⟩
  · intro initial hSO hBT hinj
    exact RealExecution.source_correct swapped p s o dtype out_dtype initial hSO hBT hinj
  · intro initial hSO hBT hinj
    exact RealSurface.source_realizes swapped p s o dtype out_dtype initial hSO hBT hinj
  · intro A hd hc initial hSO hBT hinj hb hu
    exact RealIO.source_correct swapped A hd hc p s o dtype out_dtype initial hSO hBT hinj hb hu

specification reversed_cumsum_original_real (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    Spec.Real (Implements (kernel Bool.false dtype out_dtype p s o) p s o) :=
  source_implements Bool.false dtype out_dtype p s o

specification reversed_cumsum_candidate_real (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    Spec.Real (Implements (kernel Bool.true dtype out_dtype p s o) p s o) :=
  source_implements Bool.true dtype out_dtype p s o

end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.Correctness
