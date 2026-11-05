import bench.tritonbench_g.rope_backward_transform.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Complete in-place Q/K IO with independently strided coefficient tables.
IO.lean separates all result reads from the strictly smaller writable halves. -/
def ropeIO (r : Request) := LogicalIO.view ideal id r

/-- Fixed-coefficient input adjoint. The last column of an odd dimension is
unchanged; no unit-circle or inverse-rotation assumption is needed. -/
def formula (r : Request) (xs : LogicalIO.Inputs r)
    (key : Bool) (row head col : Nat) : ℝ :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    let a := LogicalIO.query xs key row head j
    let b := LogicalIO.query xs key row head (j+r.half)
    let c := LogicalIO.coefficient xs Bool.false (row%r.sequence) j
    let s := LogicalIO.coefficient xs Bool.true (row%r.sequence) j
    if col<r.half then a*c+b*s else b*c-a*s
  else LogicalIO.query xs key row head col

/-- Both gradient tensors, with genuine logical result coordinates. -/
def mathematical (r : Request) (xs : LogicalIO.Inputs r) : LogicalIO.Outputs r where
  query := fun row head col => formula r xs Bool.false row.val head.val col.val
  key := fun row head col => formula r xs Bool.true row.val head.val col.val

/-- Full grid, both gradient outputs, odd tails and exact exterior frame. -/
specification rope_backward_kernel_correct (r : Request) :
    VeriTile.Spec.Real (ropeIO r ⊨ mathematical r) := by
  exact LogicalIO.logical_correct r

#specNonCircular formula avoiding [Source.kernel,grid,scalarResult,FP.Structural.exec]
#axiomsClean rope_backward_kernel_correct
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
