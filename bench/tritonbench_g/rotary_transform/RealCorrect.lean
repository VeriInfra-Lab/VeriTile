import bench.tritonbench_g.rotary_transform.Proofs.SourceExports
import bench.tritonbench_g.rotary_transform.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Complete selected FP32 grid, both signed metadata storage widths and all
original flags. The .real channel is the erased FP32 carrier. Inputs and the
pre-existing nonrotary suffix are explicit; only the rotary prefix is writable.
The result window still covers every logical output column. -/
def realIO (c : Config) (cumulative64 offsets64 : Bool) :=
  PublicIO.logicalIO c cumulative64 offsets64

/-- Independent rotation over logical tensor coordinates. Identity coefficient
fallback, conjugation and the unchanged input suffix are explicit. -/
def rotation (c : Config) (xs : PublicIO.RotationInputs ℝ c.shape)
    (batch : Fin c.shape.batches) (row : Fin c.shape.maximum)
    (head : Fin c.shape.heads) (col : Fin c.shape.width) : ℝ :=
  if col.val<2*c.shape.half then
    let pair := c.shape.pairOf col.val
    let second := c.shape.second col.val
    let token := c.start batch.val+row.val
    let a := xs.inputAt c token head.val (c.shape.dim pair Bool.false)
    let b := xs.inputAt c token head.val (c.shape.dim pair Bool.true)
    let cache := row.val+c.shift batch.val<c.shape.coefficientRows
    let co := if cache then xs.cosineAt c (row.val+c.shift batch.val) pair else 1
    let sn := if cache then xs.sineAt c (row.val+c.shift batch.val) pair else 0
    let signed := if c.shape.conjugate then -sn else sn
    if second then a*signed+b*co else a*co-b*signed
  else xs.inputAt c (c.start batch.val+row.val) head.val col.val

/-- Metadata and suffix agreement are logical input conditions. This shared IO
judgment proves actual source access safety, complete execution, all output
reads and the exact frame outside the rotary prefix. -/
specification rotary_transform_real (c : Config) (valid : c.Valid) (cu64 off64 : Bool) :
    Spec.Real ((realIO c cu64 off64) ⊨{PublicIO.LogicalReady c} rotation c) :=
  (PublicIO.logical_real_iff c cu64 off64).mpr (IOResults.real c valid cu64 off64)

#specNonCircular rotation avoiding [TypedGrid.run,ProgramModel.result,source,sourcePair,pairValues]
#axiomsClean rotary_transform_real
#stmtSurfaceSubset rotary_transform_real ⊆ [Spec.Real,realIO,rotation,LogicalDomainIO.ImplementsOn,
  PublicIO.LogicalReady,Config,Config.Valid,PublicIO.RotationInputs,Config.shape,Shape.batches,Shape.maximum,Shape.heads,Shape.width,ChannelValues,PublicIO.inputs,PublicIO.outputs,ReadProgramIO.ImplementsOn,PublicIO.io,realOps]
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked
