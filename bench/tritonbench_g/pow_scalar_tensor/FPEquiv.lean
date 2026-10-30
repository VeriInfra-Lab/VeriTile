import bench.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Kernel
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- No floating transformation occurs: only an integer index product changes. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (c : Parameters) : Program := ⟨kernel c,[]⟩
def commuted (c : Parameters) : Program := ⟨commutedKernel c,[]⟩

/-- Both actual shared sources succeed and every typed memory cell agrees,
for arbitrary interpretations of unchanged scalar primitives. This is an
integer-index commutation, without a power law, finite-operand domain, or
native-to-software numerical conformance assumption. -/
specification index_commuted_equiv (c : Parameters) : original c ≡[assumptions] commuted c := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · change (((kernel c).inputs,(kernel c).outputs),([] : List Site)) =
      (((commutedKernel c).inputs,(commutedKernel c).outputs),[])
    cases branch : c.oneTile <;>
      simp only [kernel,commutedKernel,branch,Bool.false_eq_true,if_false,if_true] <;> rfl
  intro α _ M D _ s _
  exact commuted_memory (FP.ProfiledRewrite.engine M) c s


#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.FPEquiv
