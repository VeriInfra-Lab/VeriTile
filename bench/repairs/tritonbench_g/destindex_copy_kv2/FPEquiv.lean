import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.ReaderFP
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Rewrite
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section
set_option autoImplicit false

def original (p : Layout) : Program := ⟨kernel p, []⟩
def indexCommuted (p : Layout) : Program := ⟨indexCommutedKernel p, []⟩
def assumptions : VeriTile.Spec.Assumptions FP.GuardedFragment := []

/-- The existing exact integer rewrite preserves successful typed execution
and every memory cell; no floating arithmetic or finite-lane premise is used. -/
specification index_commuted_equiv (p : Layout) :
    original p ≡[assumptions] indexCommuted p :=
  ReaderProof.index_commuted_equiv p


#stmtSurfaceSubset index_commuted_equiv ⊆
  [original, indexCommuted, assumptions, Layout, Spec.Equiv, Program,
   FP.ProfiledMemory.instProgramSyntaxProgram]
#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.FPEquiv
