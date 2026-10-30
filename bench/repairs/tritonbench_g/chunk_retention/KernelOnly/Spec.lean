import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section

/-- Independent retention/VJP equations for all seven output channels. Saved
histories are read from logical inputs; no state, execution or source recurrence
is reconstructed to define this mathematical target. -/
def realFormula (a : Call) (p : Nat→Nat) (xs : Inputs ℝ a.shape) : Outputs ℝ a.shape :=
  fun port => match port with
  | ⟨0,_⟩ => fun j => MathTarget.state a.shape (logicalInputs 0 a xs) Bool.true (p 2)
      (j.1.val*64) (p 0*a.shape.BK+j.2.1.val) (p 1*a.shape.BV+j.2.2.1.val)
  | ⟨1,_⟩ => fun j => MathTarget.finalState a.shape (logicalInputs 0 a xs) Bool.true (p 2)
      (p 0*a.shape.BK+j.1.val) (p 1*a.shape.BV+j.2.1.val)
  | ⟨2,_⟩ => fun j => ScalarSource.outputTile a.shape (logicalInputs 0 a xs) a.scale (p 2) (p 1)
      j.1.val (p 0*a.shape.BV+j.2.1.val) (history 0 a.shape xs Bool.false (p 2) (p 1))
  | ⟨3,_⟩ => fun j => MathTarget.future a.shape (logicalInputs 0 a xs) a.scale (p 2)
      (a.shape.boundary (a.shape.chunks-j.1.val))
      (p 0*a.shape.BK+j.2.1.val) (p 1*a.shape.BV+j.2.2.1.val)
  | ⟨4,_⟩ => fun j => ScalarSource.queryTile a.shape (logicalInputs 0 a xs) a.scale (p 2) (p 1)
      j.1.val (p 0*a.shape.BK+j.2.1.val) (history 0 a.shape xs Bool.false (p 2) (p 1))
  | ⟨5,_⟩ => fun j => ScalarSource.keyTile a.shape (logicalInputs 0 a xs) a.scale (p 2) (p 1)
      j.1.val (p 0*a.shape.BK+j.2.1.val) (history 0 a.shape xs Bool.true (p 2) (p 1))
  | ⟨6,_⟩ => fun j => ScalarSource.valuePartial a.shape (logicalInputs 0 a xs) a.scale
      (p 0) (p 2) (p 1) j.2.1.val (j.1.val*a.shape.BV+j.2.2.1.val)
      (history 0 a.shape xs Bool.true (p 2) (p 1))

/-- Independent scalar FMA reference with the original casts, final scale and
value-block order. This describes native observation, not native=Real equality. -/
def nativeFormula {α : Type} (ops : Ops α) (literal : ℝ→α) (a : Call)
    (p : Nat→Nat) (xs : Inputs α a.shape) : Outputs α a.shape :=
  fun port => match port with
  | ⟨0,_⟩ => fun j => ops.cast (Reference.history ops a.shape (logicalInputs ops.zero a xs)
      (p 2) j.1.val (p 0*a.shape.BK+j.2.1.val) (p 1*a.shape.BV+j.2.2.1.val))
  | ⟨1,_⟩ => fun j => Reference.finalState ops a.shape (logicalInputs ops.zero a xs) (p 2)
      (p 0*a.shape.BK+j.1.val) (p 1*a.shape.BV+j.2.1.val)
  | ⟨2,_⟩ => fun j => Body.output ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (p 2) (p 1) (history ops.zero a.shape xs Bool.false (p 2) (p 1))
      j.1.val (p 0*a.shape.BV+j.2.1.val)
  | ⟨3,_⟩ => fun j => ops.cast (Reference.reverseCarry ops a.shape (logicalInputs ops.zero a xs)
      (literal a.scale) (p 2) j.1.val (p 0*a.shape.BK+j.2.1.val) (p 1*a.shape.BV+j.2.2.1.val))
  | ⟨4,_⟩ => fun j => Body.dq ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (p 2) (p 1) (history ops.zero a.shape xs Bool.false (p 2) (p 1))
      j.1.val (p 0*a.shape.BK+j.2.1.val)
  | ⟨5,_⟩ => fun j => Body.dk ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (p 2) (p 1) (history ops.zero a.shape xs Bool.true (p 2) (p 1))
      j.1.val (p 0*a.shape.BK+j.2.1.val)
  | ⟨6,_⟩ => fun j => Body.dvPartial ops a.shape (logicalInputs ops.zero a xs) (literal a.scale)
      (p 0) (p 2) (p 1) (history ops.zero a.shape xs Bool.true (p 2) (p 1))
      j.2.1.val (j.1.val*a.shape.BV+j.2.2.1.val)

#specNonCircular realFormula avoiding
  [FP.Structural.run,FP.Structural.exec,KernelPacket.realTarget,ForwardHSource.update,BackwardQKVSource.queryValue]
#specNonCircular nativeFormula avoiding
  [FP.Structural.run,FP.Structural.exec,KernelPacket.nativeTarget,KernelStaging.gradientDot]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
