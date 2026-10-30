import bench.tritonbench_g.rotary_transform_ops.Proofs.LogicalResult
import VeriTile.Meta.Specification

/-! Independent mathematical rotation for the complete original launch grid.
IO.lean exposes FP32 channels, signed metadata, aliasing, suffix entry values,
result windows and the exact writable prefix. Operational facts are in Proofs. -/
namespace VeriTile.Bench.Repairs.RotaryTransformOps
open VeriTile.Triton FP.Structural Interface
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Independent rotation at each logical tensor coordinate. The identity
coefficient fallback, conjugation and unchanged suffix are explicit. -/
def rotationAt (c : Config) (xs : RotationInputs ℝ c.shape)
    (batch row head col : Nat) : ℝ :=
  if col<2*c.shape.half then
    let pair := c.shape.pairOf col
    let second := c.shape.second col
    let token := c.start batch+row
    let a := xs.inputAt c token head (c.shape.dim pair Bool.false)
    let b := xs.inputAt c token head (c.shape.dim pair Bool.true)
    let cache := row+c.shift batch<c.shape.coefficientRows
    let co := if cache then xs.cosineAt c (row+c.shift batch) pair else 1
    let sn := if cache then xs.sineAt c (row+c.shift batch) pair else 0
    let signed := if c.shape.conjugate then -sn else sn
    if second then a*signed+b*co else a*co-b*signed
  else xs.inputAt c (c.start batch+row) head col

def rotation (c : Config) (xs : RotationInputs ℝ c.shape)
    (batch : Fin c.shape.batches) (row : Fin c.shape.maximum)
    (head : Fin c.shape.heads) (col : Fin c.shape.width) : ℝ :=
  rotationAt c xs batch.val row.val head.val col.val

/-- Every output column is the independent rotation formula, including the
unchanged suffix. The shared judgment also proves actual source access safety,
successful full-grid execution, and the exact exterior-cell frame. -/
specification rotary_transform_ops_correctness (c : Config) (valid : c.Valid)
    (cumulative64 offsets64 : Bool) :
    Spec.Real ((logicalIO c cumulative64 offsets64) ⊨{LogicalReady c} rotation c) :=
  (logical_real_iff c cumulative64 offsets64).mpr
    (IOComplete.real_implements c valid cumulative64 offsets64)

#specNonCircular rotation avoiding
  [FP.Structural.exec,TypedGrid.run,ProgramModel.result,source,sourcePair,pairValues]
#stmtSurfaceSubset rotary_transform_ops_correctness ⊆
  [Spec.Real,Config,Config.Valid,logicalIO,LogicalDomainIO.ImplementsOn,LogicalReady,rotation,RotationInputs,Config.shape,Shape.batches,Shape.maximum,Shape.heads,Shape.width,ChannelValues,inputs,outputs,ReadProgramIO.ImplementsOn,io,realOps,Inputs,Outputs]
#axiomsClean rotary_transform_ops_correctness
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransformOps
