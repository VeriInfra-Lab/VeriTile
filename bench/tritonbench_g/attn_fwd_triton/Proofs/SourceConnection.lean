import bench.tritonbench_g.attn_fwd_triton.Proofs.SourceFragments

/- Exact execution connection between the repair quotation and its proof view.
The view is the complete source projection, with no branch factoring. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified
open VeriTile.Triton
set_option maxRecDepth 32000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def actualCode (BM BN BD : Nat) (p : Parameters) (r : Bool) : List Stmt :=
  (source BM BN BD p r).eraseDType.toAlgKernel.body

theorem code_run (BM BN BD : Nat) (p : Parameters) (r : Bool) (s : BlockState) :
    stepStmts (code BM BN BD p r) s=stepStmts (actualCode BM BN BD p r) s := rfl

theorem actual_execution (BM BN BD : Nat) (p : Parameters) (r : Bool)
    (s t : BlockState) (h : stepStmts (code BM BN BD p r) s=some t) :
    stepStmts (actualCode BM BN BD p r) s=some t := by
  rw [←code_run]
  exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified
