import bench.repairs.tritonbench_g.triton_mul2.Kernels
import bench.repairs.tritonbench_g.triton_mul2.Proofs.FPFacts
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

/-! Equivalence of the actual FP32 kernels, for both output modes. The only
numerical transformation is strict FP32 multiplication commutation on active loaded operands. -/
namespace VeriTile.Bench.Repairs.TritonMul2.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section
abbrev Rules := FP.ScalarArithmetic.Rules

/-- Finiteness belongs to the observed rewrite site, not Python admission.
Inactive lanes may contain arbitrary, including nonfinite, load values. -/
abbrev multiplicationSite := Rewrite.multiplicationSite

def original (X O : RegionName) (n B : Nat) : Program where
  kernel := sourceKernel X O n B
  domain := [multiplicationSite X O n B]

def commuted (X O : RegionName) (n B : Nat) : Program :=
  { original X O n B with kernel := commutedSourceKernel X O n B }

/-- Both complete sources succeed and agree on every typed memory cell.
The proof supports equal input/output regions, not just fresh output. -/
specification mul2_commuted_equiv (X O : RegionName) (n B : Nat) :
    original X O n B ≡[([] : Spec.Assumptions FP.GuardedFragment)] commuted X O n B := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM s hd
  exact ⟨(Rewrite.originalExecution M X O n B s).val,
    (Rewrite.commutedExecution M X O n B s).val,
    (Rewrite.originalExecution M X O n B s).property,
    (Rewrite.commutedExecution M X O n B s).property,
    Rewrite.commuted_memory M D hM X O n B s (hd _ (by simp [original]))⟩

def inplaceOriginal (X : RegionName) (n B : Nat) : Program :=
  { original X X n B with kernel := inplaceSourceKernel X n B }

def inplaceCommuted (X : RegionName) (n B : Nat) : Program :=
  { original X X n B with kernel := inplaceCommutedSourceKernel X n B }

specification mul2_inplace_commuted_equiv (X : RegionName) (n B : Nat) :
    inplaceOriginal X n B ≡[([] : Spec.Assumptions FP.GuardedFragment)] inplaceCommuted X n B :=
  mul2_commuted_equiv X X n B

#print_fp_assumptions mul2_commuted_equiv
#print_fp_assumptions mul2_inplace_commuted_equiv
#axiomsClean mul2_commuted_equiv
#axiomsClean mul2_inplace_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonMul2.FPEquiv
