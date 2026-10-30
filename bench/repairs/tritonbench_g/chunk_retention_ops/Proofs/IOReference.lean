import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOReverseLocality
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOLocalReadback

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000

/-- Independent input-only tile recurrences already proved for the source.
This helper is used to transport results; public formulas are the separate
scalar and dense formulas in KernelOnly/Formula.lean. -/
def sourceValues (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : KernelPacket.Memory α) :
    Outputs α a.shape :=
  let c:=a.shape
  let p:=parameters c
  let cast:=M.cast (some .fp32) .real .real
  fun port => match port with
  | ⟨0,_⟩ => fun j => cast (KernelForward.values M D c a.pid m a.inputState j.1.val j.2)
  | ⟨1,_⟩ => fun j => cast (KernelForward.values M D c a.pid m a.inputState c.chunks j)
  | ⟨2,_⟩ => KernelOutput.outputValue M D c a.scale a.pid m
  | ⟨3,_⟩ => fun j => cast (KernelBackward.values M D p a.pid m a.scale a.inputState j.1.val j.2)
  | ⟨4,_⟩ => fun j => cast (KernelBackward.values M D p a.pid m a.scale a.inputState c.chunks j)
  | ⟨5,_⟩ => fun j => cast (KernelGradient.gradQueryResult M D p a.pid m a.scale
      (KernelGradient.gradStates M D p a.pid m a.scale c.valueBlocks) j)
  | ⟨6,_⟩ => fun j => cast (KernelGradient.gradKeyResult M D p a.pid m a.scale
      (KernelGradient.gradStates M D p a.pid m a.scale c.valueBlocks) j)
  | ⟨7,_⟩ => fun j => cast (KernelGradient.gradValue M D p a.pid m a.scale j.1.val j.2)

/-- Every finite active output is an actual typed source store. The local dV
proof holds for arbitrary program coordinates, including heads outside the
host grid. -/
theorem reference_read (M : Algebra α) (D : DotPrimitive α) (a : Call) (m : KernelPacket.Memory α)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a port j) :
    KernelPacket.reference M D a m (outputs a.shape)[port].buf (outputOffset a port j)=
      Cell.mk (outputs a.shape)[port].dtype (sourceValues M D a m port j) := by
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  · rcases active with ⟨stage,active⟩
    simpa only [KernelPacket.reference,stage,outputs,outputOffset,sourceValues] using
      KernelForward.state_read_history M D a.shape a.pid m m a.inputState a.outputState j.1.val j.2 j.1.isLt active
  · rcases active with ⟨stage,final,active⟩
    simpa only [KernelPacket.reference,stage,final,outputs,outputOffset,sourceValues] using
      KernelForward.state_read_final M D a.shape a.pid m m a.inputState j active
  · rcases active with ⟨stage,active⟩
    simpa only [KernelPacket.reference,stage,outputs,outputOffset,sourceValues] using
      KernelOutput.memory_read M D a.shape a.scale a.pid m m j active
  · rcases active with ⟨stage,active⟩
    simpa only [KernelPacket.reference,stage,outputs,outputOffset,sourceValues] using
      KernelBackward.state_read_history M D (parameters a.shape) a.pid m m a.scale a.inputState a.outputState
        j.1.val j.2 j.1.isLt active
  · rcases active with ⟨stage,final,active⟩
    simpa only [KernelPacket.reference,stage,final,outputs,outputOffset,sourceValues] using
      KernelBackward.state_read_final M D (parameters a.shape) a.pid m m a.scale a.inputState j active
  · rcases active with ⟨stage,active⟩
    simpa only [KernelPacket.reference,stage,outputs,outputOffset,sourceValues] using
      KernelGradient.gradient_read_query M D (parameters a.shape) a.pid m m a.scale j active
  · rcases active with ⟨stage,active⟩
    simpa only [KernelPacket.reference,stage,outputs,outputOffset,sourceValues] using
      KernelGradient.gradient_read_key M D (parameters a.shape) a.pid m m a.scale j active
  · rcases active with ⟨stage,active⟩
    simpa only [KernelPacket.reference,stage,outputs,outputOffset,sourceValues] using
      KernelGradient.gradient_read_value_local M D (parameters a.shape) a.pid (a.shape.B*a.shape.H)
        m m a.scale j.1.val j.2 j.1.isLt active

theorem source_values_readEq (M : Algebra α) (D : DotPrimitive α) (a : Call)
    (m n : KernelPacket.Memory α) (reads : ReadEq a m n)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a port j) :
    sourceValues M D a m port j=sourceValues M D a n port j := by
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      (forward_values_readEq M D a active.1 m n reads j.1.val) j.2)
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      (forward_values_readEq M D a active.1 m n reads a.shape.chunks) j)
  · exact congrFun (output_value_readEq M D a active.1 m n reads) j
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      (backward_values_readEq M D a active.1 m n reads j.1.val) j.2)
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      (backward_values_readEq M D a active.1 m n reads a.shape.chunks) j)
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      ((gradient_loads_readEq M a active.1 m n reads).queryResult D a.shape.valueBlocks) j)
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      ((gradient_loads_readEq M a active.1 m n reads).keyResult D a.shape.valueBlocks) j)
  · exact congrArg (M.cast (some .fp32) .real .real) (congrFun
      ((gradient_loads_readEq M a active.1 m n reads).valueResult D j.1.val) j.2)

/-- The finite public outputs cover the complete original write footprint. -/
theorem footprint_covered (a : Call) (r : RegionName) (off : Nat) (written : KernelPacket.Footprint a r off) :
    ∃ port,∃ j : TileIndex (outputs a.shape)[port].shape,
      outputActive a port j ∧ r=(outputs a.shape)[port].buf ∧ off=outputOffset a port j := by
  cases stage : a.stage with
  | forwardState =>
    simp only [KernelPacket.Footprint,stage] at written
    change KernelForward.Footprint a.shape a.pid a.outputState r off at written
    rcases written with ⟨hr,n,hn,i,hi,ho⟩|⟨hr,hf,i,hi,ho⟩
    · exact ⟨0,(⟨n,hn⟩,i),⟨stage,hi⟩,hr,ho⟩
    · exact ⟨1,i,⟨stage,hf,hi⟩,hr,ho⟩
  | forwardOutput =>
    simp only [KernelPacket.Footprint,stage] at written
    change KernelOutput.Footprint a.shape a.pid r off at written
    rcases written with ⟨hr,i,hi,ho⟩
    exact ⟨2,i,⟨stage,hi⟩,hr,ho⟩
  | backwardState =>
    simp only [KernelPacket.Footprint,stage] at written
    change KernelPacket.backwardFootprint a r off at written
    rcases written with ⟨hr,n,hn,i,hi,ho⟩|⟨hr,hf,i,hi,ho⟩
    · exact ⟨3,(⟨n,hn⟩,i),⟨stage,hi⟩,hr,ho⟩
    · exact ⟨4,i,⟨stage,hf,hi⟩,hr,ho⟩
  | backwardGradient =>
    simp only [KernelPacket.Footprint,stage] at written
    change VeriTile.Bench.Repairs.ChunkLinearAttn.GradientFootprint (parameters a.shape) a.pid r off at written
    rcases written with ⟨hr,i,hi,ho⟩|⟨hr,vb,hv,i,hi,ho⟩
    · rcases hr with hr|hr
      · exact ⟨5,i,⟨stage,hi⟩,hr,ho⟩
      · exact ⟨6,i,⟨stage,hi⟩,hr,ho⟩
    · exact ⟨7,(⟨vb,hv⟩,i),⟨stage,hi⟩,hr,ho⟩

#axiomsClean reference_read
#axiomsClean source_values_readEq
#axiomsClean footprint_covered
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
