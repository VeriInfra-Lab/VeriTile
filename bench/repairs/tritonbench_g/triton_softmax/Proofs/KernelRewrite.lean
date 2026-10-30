import bench.repairs.tritonbench_g.triton_softmax.Kernels
import bench.tritonbench_g.triton_softmax.Proofs.IndexRewrite

namespace VeriTile.Bench.Repairs.TritonSoftmax.KernelRewrite
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def program (k : ComputeKernel) : Program := ⟨k,[]⟩

/-- A genuine source candidate: exact integer commutation in the input
row address. The complete floating program remains in its original context. -/
theorem equivalent (Y X : RegionName) (ib ob ir ic orow oc N B : Nat) :
    program (softmax_kernel Y X ib ob ir ic orow oc N B) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      program (indexCommutedKernel Y X ib ob ir ic orow oc N B) := by
  equiv_decompose; fp_prove

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (Y X : RegionName) (ib ob ir ic orow oc N B : Nat) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (softmax_kernel Y X ib ob ir ic orow oc N B) s=
      FP.Structural.exec (engine algebra) (indexCommutedKernel Y X ib ob ir ic orow oc N B) s := by
  let original := (softmax_kernel Y X ib ob ir ic orow oc N B).surfaceBody
  let candidate := (indexCommutedKernel Y X ib ob ir ic orow oc N B).surfaceBody
  -- The second assignment is the input row pointer; all other statements are shared.
  have h : Contextual algebra s (original.take 1) (original.drop 2)
      (original.getD 1 (.ifThen (.constBool Bool.false) []))
      (candidate.getD 1 (.ifThen (.constBool Bool.false) [])) := by
    fp_prove
  exact h

#axiomsClean equivalent
#axiomsClean execution
end VeriTile.Bench.Repairs.TritonSoftmax.KernelRewrite
