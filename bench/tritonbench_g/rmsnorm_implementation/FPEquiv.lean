import bench.tritonbench_g.rmsnorm_implementation.Proofs.FPInterface

/-! Successful full-memory equivalence of the canonical FP32 programs. Only a
natural-number index multiplication is commuted. Both complete loops, source
precision, masks, reduction order and arbitrary undefined payloads are retained.
The separate observed native graph and compiler boundary are in Misc/Auxiliary.lean. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.FPEquiv
open VeriTile Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := FPInterface.assumptions

def original (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) : Program :=
  FPInterface.original sxb sxm sxk sw sob som sok n B eps
def commuted (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) : Program :=
  FPInterface.commuted sxb sxm sxk sw sob som sok n B eps

/-- Both complete typed programs succeed and agree on every memory cell. -/
specification rmsnorm_index_commuted_equiv (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) :
    original sxb sxm sxk sw sob som sok n B eps ≡[assumptions]
      commuted sxb sxm sxk sw sob som sok n B eps :=
  FPInterface.rmsnorm_index_commuted_equiv sxb sxm sxk sw sob som sok n B eps

#print_fp_assumptions rmsnorm_index_commuted_equiv
#axiomsClean rmsnorm_index_commuted_equiv
#stmtSurfaceSubset rmsnorm_index_commuted_equiv ⊆
  [original,commuted,assumptions,Spec.Equiv,Program,instProgramSyntaxProgram]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.FPEquiv
