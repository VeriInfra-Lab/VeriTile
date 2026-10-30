import bench.tritonbench_g.chunk_retention_ops.IO
import bench.tritonbench_g.chunk_retention_ops.Sources.ScalarOutput

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

/-- Independent dense retention/adjoint formulas. Histories are genuine inputs
to the standalone output and QKV stages, not assumed results of another stage. -/
def mathOutputs (a : Call) (data : Whole001.Inputs ℝ) (saved future : Tile ℝ) : Outputs ℝ a.shape :=
  let c:=a.shape
  fun port => match port with
  | ⟨0,_⟩ => fun j => MathTarget.state c data Bool.true a.pid.z (c.boundary j.1.val)
      (a.pid.x*c.BK+j.2.1.val) (a.pid.y*c.BV+j.2.2.1.val)
  | ⟨1,_⟩ => fun j => MathTarget.finalState c data Bool.true a.pid.z
      (a.pid.x*c.BK+j.1.val) (a.pid.y*c.BV+j.2.1.val)
  | ⟨2,_⟩ => fun j => ScalarSource.outputTile c data a.scale a.pid.z a.pid.y j.1.val
      (a.pid.x*c.BV+j.2.1.val) saved
  | ⟨3,_⟩ => fun j => MathTarget.future c data a.scale a.pid.z (c.boundary (c.chunks-j.1.val))
      (a.pid.x*c.BK+j.2.1.val) (a.pid.y*c.BV+j.2.2.1.val)
  | ⟨4,_⟩ => fun j => MathTarget.initialGradient c data a.scale a.pid.z
      (a.pid.x*c.BK+j.1.val) (a.pid.y*c.BV+j.2.1.val)
  | ⟨5,_⟩ => fun j => ScalarSource.queryTile c data a.scale a.pid.z a.pid.y j.1.val
      (a.pid.x*c.BK+j.2.1.val) saved
  | ⟨6,_⟩ => fun j => ScalarSource.keyTile c data a.scale a.pid.z a.pid.y j.1.val
      (a.pid.x*c.BK+j.2.1.val) future
  | ⟨7,_⟩ => fun j => ScalarSource.valuePartial c data a.scale a.pid.x a.pid.z a.pid.y j.2.1.val
      (j.1.val*c.BV+j.2.2.1.val) future

/-- Existing ordered native scalar reference: all casts, FMA orders and the
dK materialization boundary are retained. This is not equality to exact Real. -/
def scalarOutputs {α : Type} (ops : Ops α) (scale : α) (a : Call)
    (data : Whole001.Inputs α) (saved future : Tile α) : Outputs α a.shape :=
  let c:=a.shape
  fun port => match port with
  | ⟨0,_⟩ => fun j => ops.cast (Reference.history ops c data a.pid.z j.1.val
      (a.pid.x*c.BK+j.2.1.val) (a.pid.y*c.BV+j.2.2.1.val))
  | ⟨1,_⟩ => fun j => Reference.finalState ops c data a.pid.z
      (a.pid.x*c.BK+j.1.val) (a.pid.y*c.BV+j.2.1.val)
  | ⟨2,_⟩ => fun j => Body.output ops c data scale a.pid.z a.pid.y
      saved j.1.val (a.pid.x*c.BV+j.2.1.val)
  | ⟨3,_⟩ => fun j => ops.cast (Reference.reverseCarry ops c data scale a.pid.z j.1.val
      (a.pid.x*c.BK+j.2.1.val) (a.pid.y*c.BV+j.2.2.1.val))
  | ⟨4,_⟩ => fun j => ops.cast (Reference.reverseCarry ops c data scale a.pid.z c.chunks
      (a.pid.x*c.BK+j.1.val) (a.pid.y*c.BV+j.2.1.val))
  | ⟨5,_⟩ => fun j => Body.dq ops c data scale a.pid.z a.pid.y
      saved j.1.val (a.pid.x*c.BK+j.2.1.val)
  | ⟨6,_⟩ => fun j => Body.dk ops c data scale a.pid.z a.pid.y
      future j.1.val (a.pid.x*c.BK+j.2.1.val)
  | ⟨7,_⟩ => fun j => Body.dvPartial ops c data scale a.pid.x a.pid.z a.pid.y
      future j.2.1.val (j.1.val*c.BV+j.2.2.1.val)

/-- Independent dense mathematical outputs on the finite input channels. -/
def mathematical (a : Call) (xs : Inputs ℝ a.shape) : Outputs ℝ a.shape :=
  mathOutputs a (logicalInputs 0 a xs) (history 0 a.shape xs Bool.false) (history 0 a.shape xs Bool.true)

/-- Independent ordered scalar outputs on the finite input channels. -/
def scalarFormula {α : Type} (ops : Ops α) (scale : α) (a : Call)
    (xs : Inputs α a.shape) : Outputs α a.shape :=
  scalarOutputs ops scale a (logicalInputs ops.zero a xs)
    (history ops.zero a.shape xs Bool.false) (history ops.zero a.shape xs Bool.true)

/-- Native FP32 specialization of the independent ordered scalar reference. -/
def nativeFormula (isa : ISA) (literal : ℝ→KernelNative.Word) (a : Call)
    (xs : Inputs KernelNative.Word a.shape) : Outputs KernelNative.Word a.shape :=
  scalarFormula (wordOps isa) (literal a.scale) a xs

#specNonCircular mathematical avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelPacket.source]
#specNonCircular nativeFormula avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelPacket.source]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
