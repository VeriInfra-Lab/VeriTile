import bench.tritonbench_g.kldiv_ops.Proofs.RealGrid
import bench.tritonbench_g.kldiv_ops.Proofs.FPGraph

namespace VeriTile.Bench.Repairs.KldivOps.RealExecution
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

def logicalMemory (prediction target : Nat→ℝ) : Program.Memory ℝ :=
  fun region i=>.mk .real (if region="Y" then prediction i else target i)

theorem row_ideal (flag : Bool) (eps : ℝ) (cols width : Nat) (hb : 0<width)
    (prediction target : Nat→ℝ) :
    FPGraph.row algebra cols width flag eps prediction target=rowLoss flag eps prediction target cols := by
  let p : Parameters := ⟨"Y","T","O",1,cols,width⟩
  have graph := FPGraph.row_binding algebra p flag 1 0 eps (logicalMemory prediction target) (by decide)
  have math := reduced_row p flag 1 0 eps (logicalMemory prediction target) hb (by decide)
  have values := congrArg (fun c : Cell ℝ=>c.read .real) (graph.symm.trans math)
  simpa [p,logicalMemory] using values

theorem backward_scalar_ideal (flag : Bool) (target : ℝ) :
    Backward.scalar algebra flag target=backward flag target := by
  cases flag <;> simp [Backward.scalar,algebra,backward]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.RealExecution
