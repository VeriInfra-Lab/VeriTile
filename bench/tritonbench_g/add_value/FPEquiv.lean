import bench.tritonbench_g.add_value.Kernels
import bench.tritonbench_g.add_value.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- Selected typed FP32 kernel source. -/
def source (n : Nat) : ComputeKernel := originalKernel n

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Actual operands after the original FP32 load prefix. Finiteness is required
only where the source mask is true; masked-off values are unrestricted.
This is a rewrite-site condition, not a Python input guard. -/
abbrev additionSite (n : Nat) : Site := Rewrite.additionSite n

/-- The exact shared source retains its explicit FP32 load, arithmetic and
store tags; the rewrite acts on that source, without dtype erasure. -/
def original (n : Nat) : Program where
  kernel := originalKernel n
  domain := [additionSite n]

/-- Local `y = 10; output = y + x`, with the same actual rewrite-site domain. -/
def commuted (n : Nat) : Program :=
  { original n with kernel := commutedKernel n }

/-- Both displayed sources execute successfully and agree on every typed
memory cell by strict FP32 addition commutation at the existing operand site. The candidate's
extra local register `y` is private and need not occur in the original. -/
specification add_value_commuted_equiv (n : Nat) :
    original n ≡[([] : Spec.Assumptions FP.GuardedFragment)] commuted n := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM s hd
  exact ⟨(Rewrite.originalExecution M n s).val,
    (Rewrite.commutedExecution M n s).val,
    (Rewrite.originalExecution M n s).property,
    (Rewrite.commutedExecution M n s).property,
    Rewrite.commuted_memory M D hM n s (hd _ (by simp [original]))⟩

#print_fp_assumptions add_value_commuted_equiv
#axiomsClean add_value_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.FPEquiv
