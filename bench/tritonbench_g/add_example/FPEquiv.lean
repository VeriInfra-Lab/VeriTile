import bench.tritonbench_g.add_example.Kernels
import bench.tritonbench_g.add_example.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddExample.Contiguous.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules
abbrev source (n : Nat) : ComputeKernel := originalKernel n

/-- Only actually observed lanes require finite operands after the FP32 loads.
This is the transformation domain, separate from Python input admission. -/
abbrev additionSite (n : Nat) : Site := Rewrite.additionSite n blockSize

def original (n : Nat) : Program where
  kernel := originalKernel n
  domain := [additionSite n]

def commuted (n : Nat) : Program :=
  { original n with kernel := commutedKernel n }

/-- Both complete sources succeed and agree on every typed memory cell.
The only numerical transformation is the shared strict FP32 addition commutation. -/
specification add_commuted_equiv (n : Nat) :
    original n ≡[([] : Spec.Assumptions FP.GuardedFragment)] commuted n := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM s hd
  exact ⟨(Rewrite.originalExecution M n blockSize s).val,
    (Rewrite.commutedExecution M n blockSize s).val,
    (Rewrite.originalExecution M n blockSize s).property,
    (Rewrite.commutedExecution M n blockSize s).property,
    Rewrite.commuted_memory M D hM n blockSize s (hd _ (by simp [original]))⟩

#print_fp_assumptions add_commuted_equiv
#axiomsClean add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddExample.Contiguous.FPEquiv
