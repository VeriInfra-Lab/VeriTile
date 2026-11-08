import bench.tritonbench_g.rope_transform.Proofs.LogicalReader
import bench.tritonbench_g.rope_transform.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.RopeTransform.Correct
open VeriTile.Triton FP.Structural KernelContract
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Actual FP32 source and full row schedule, with both Q/K regions explicitly
in-place and C/S read-only. IO.lean displays all channels, windows and masks. -/
@[reducible] def realIO (r : Request) := LogicalIO.view realAlgebra r

/-- Independent rotation polynomial; no source execution is part of the target. -/
def formula (a b c s : ℝ) (second : Bool) : ℝ :=
  if second then b*c+a*s else a*c-b*s

/-- Independent rotation on named logical tensors; no memory channel occurs. -/
def rotation (r : Request) (xs : LogicalIO.Inputs r)
    (key : Bool) (row head col : Nat) : ℝ :=
  let lane := if col<r.half then col else col-r.half
  let a := LogicalIO.query xs key row head lane
  let b := LogicalIO.query xs key row head (lane+r.half)
  let c := LogicalIO.coefficient xs Bool.false (row%r.sequence) lane
  let s := LogicalIO.coefficient xs Bool.true (row%r.sequence) lane
  if decide (r.half≤col) then b*c+a*s else a*c-b*s

def mathematical (r : Request) (xs : LogicalIO.Inputs r) : LogicalIO.Outputs r where
  query := fun row head col => rotation r xs Bool.false row.val head.val col.val
  key := fun row head col => rotation r xs Bool.true row.val head.val col.val

specification rope_forward_correct (r : Request) :
    Spec.Real (realIO r ⊨ mathematical r) := by
  exact LogicalIO.logical_correct r

#specNonCircular formula avoiding [kernel,scalar,Grid.run]
#stmtSurfaceSubset rope_forward_correct ⊆
  [Request,Spec.Real,realIO,LogicalProgramIO.Implements,mathematical,LogicalIO.Inputs,LogicalIO.Outputs]
#axiomsClean rope_forward_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeTransform.Correct
