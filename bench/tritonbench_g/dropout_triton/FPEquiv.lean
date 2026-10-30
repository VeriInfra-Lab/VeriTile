import bench.tritonbench_g.dropout_triton.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory

/-! The existing candidate commutes integer index multiplication. All FP32
operations remain in the same order, so equivalence has no numerical premises.
Successful execution, complete memory equality and the frame are explicit. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : VeriTile.Spec.Assumptions FP.GuardedFragment := []

def original (n : Nat) (p : ℝ) : Program where
  kernel := DropoutTriton.kernel n p
  domain := []

def indexCommuted (n : Nat) (p : ℝ) : Program where
  kernel := indexCommutedKernel n p
  domain := []

/-- No domain guard is added for masks, data values, p=1 or inactive lanes.
This is the actual index-order variant, not an identity substitution. -/
specification dropout_index_commuted_equiv (n : Nat) (p : ℝ) :
    original n p ≡[assumptions] indexCommuted n p := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM s hd
  let result := Rewrite.originalExecution M n p s
  refine ⟨result.val, result.val, result.property, ?_, rfl⟩
  exact (Rewrite.execution_equal M n p s).symm.trans result.property

#print_fp_assumptions dropout_index_commuted_equiv
#axiomsClean dropout_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.FPEquiv
