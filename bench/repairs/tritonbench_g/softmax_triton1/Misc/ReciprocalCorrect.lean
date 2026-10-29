import bench.repairs.tritonbench_g.softmax_triton1.Correct

/-! Auxiliary Real correctness of the reciprocal candidate. The original kernel correctness is in Correct. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The retained reciprocal source has the same mathematical row result. -/
def reciprocalIO := RealIO.reciprocalIO

theorem reciprocal_kernel_correctness (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) :
    Spec.Real (reciprocalIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) :=
  RealIO.reciprocal_correct out inp ib ob n B hB

#axiomsClean reciprocal_kernel_correctness

#stmtSurfaceSubset reciprocal_kernel_correctness ⊆
  [RegionName, Spec.Real, reciprocalIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Correct
