import bench.tritonbench_g.square_matrix.Proofs.KernelFP
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SquareMatrix.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open VeriTile.Torch.FP32
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (n B : Nat) : Program where
  kernel := fp32Kernel n B
  domain := []

def indexCommuted (n B : Nat) : Program :=
  { original n B with kernel := indexCommutedKernel n B }

/-- Both actual sources succeed and agree on every typed memory cell. Only
integer index multiplication is commuted; FP32 multiplication is identical.
There are no finite-value or inactive-lane numerical requirements. -/
specification square_index_equiv (n B : Nat) :
    original n B ≡[assumptions]
      indexCommuted n B := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  obtain ⟨t, executed⟩ := KernelFP.successful (FP.ProfiledRewrite.engine M)
    n B s
  exact ⟨t, t, executed,
    (KernelFP.variants_equal (FP.ProfiledRewrite.engine M) _ _ s).symm.trans executed, rfl⟩

#print_fp_assumptions square_index_equiv
#axiomsClean square_index_equiv
#stmtSurfaceSubset square_index_equiv ⊆
  [original, indexCommuted, assumptions, VeriTile.Spec.Equiv,
   FP.ProfiledMemory.Program, FP.ProfiledMemory.instProgramSyntaxProgram]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.FPEquiv
