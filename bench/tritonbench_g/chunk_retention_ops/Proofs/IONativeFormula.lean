import bench.tritonbench_g.chunk_retention_ops.Proofs.IOSharedFormula
import bench.tritonbench_g.chunk_retention_ops.Proofs.IOFormulaCongruence
import bench.tritonbench_g.chunk_retention_ops.Proofs.IOReference
import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeBackward
import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeGradient

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelNative
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 30000

private theorem forward_history_scalar (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call)
    (stage : a.stage=.forwardState) (m : RegionName→Nat→Word) (n : Nat)
    (hn : n<a.shape.chunks) (i : TileIndex [a.shape.BK,a.shape.BV])
    (hi : VeriTile.Bench.Repairs.ChunkLinearAttn.hActive (KernelReal.parameters a.shape) a.pid i) :
    S.lower (sharedOps (wordOps isa)) literal
      (model.cast (some .fp32) .real .real (KernelForward.values model (dotModel a) a.shape a.pid (input m) a.inputState n i))=
    (wordOps isa).cast (Reference.history (wordOps isa) a.shape
      (readInputs (wordOps isa).zero a m) a.pid.z n (a.pid.x*a.shape.BK+i.1.val) (a.pid.y*a.shape.BV+i.2.1.val)) := by
  let b : Call:=⟨a.shape,.forwardState,0,a.inputState,a.outputState,a.pid⟩
  have h := (KernelNative.forward_target isa literal laws a.shape a.inputState a.outputState a.pid m).1 n hn i hi
  have read := reference_read model (dotModel b) b (input m) ⟨0,by simp [outputs]⟩ (⟨n,hn⟩,i) ⟨rfl,hi⟩
  change KernelPacket.reference model (dotModel b) b (input m) "h"
    (VeriTile.Bench.Repairs.ChunkLinearAttn.hOffset (KernelReal.parameters a.shape) a.pid n i)=Cell.mk .real _ at read
  change S.lower (sharedOps (wordOps isa)) literal
    ((KernelPacket.reference model (dotModel b) b (input m) _ _).read .real)=_ at h
  rw [read] at h
  change S.lower (sharedOps (wordOps isa)) literal
      (model.cast (some .fp32) .real .real (KernelForward.values model (dotModel b) a.shape a.pid (input m) a.inputState n i))=
    (wordOps isa).cast (VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history
      (sharedOps (wordOps isa)) (KernelOutput.shape a.shape)
      (sharedInputs (KernelNative.data (wordOps isa) a.shape m a.inputState Bool.false))
      a.pid.z n (a.pid.x*a.shape.BK+i.1.val) (a.pid.y*a.shape.BV+i.2.1.val)) at h
  rw [shared_history] at h
  have same := history_congr (wordOps isa) a.shape
    (KernelNative.data (wordOps isa) a.shape m a.inputState Bool.false)
    (readInputs (wordOps isa).zero a m) a.pid.z rfl rfl rfl n
  rw [same] at h
  simpa only [dotModel,stage,b] using h

private theorem forward_final_scalar (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call)
    (stage : a.stage=.forwardState) (m : RegionName→Nat→Word)
    (i : TileIndex [a.shape.BK,a.shape.BV])
    (hi : VeriTile.Bench.Repairs.ChunkLinearAttn.hActive (KernelReal.parameters a.shape) a.pid i) :
    S.lower (sharedOps (wordOps isa)) literal
      (model.cast (some .fp32) .real .real (KernelForward.values model (dotModel a) a.shape a.pid (input m) a.inputState a.shape.chunks i))=
    Reference.finalState (wordOps isa) a.shape (readInputs (wordOps isa).zero a m) a.pid.z
      (a.pid.x*a.shape.BK+i.1.val) (a.pid.y*a.shape.BV+i.2.1.val) := by
  let b : Call:=⟨a.shape,.forwardState,0,a.inputState,Bool.true,a.pid⟩
  have h := (KernelNative.forward_target isa literal laws a.shape a.inputState Bool.true a.pid m).2 rfl i hi
  have read := reference_read model (dotModel b) b (input m) ⟨1,by simp [outputs]⟩ i ⟨rfl,rfl,hi⟩
  change KernelPacket.reference model (dotModel b) b (input m) "ht"
    (VeriTile.Bench.Repairs.ChunkLinearAttn.fOffset (KernelReal.parameters a.shape) a.pid i)=Cell.mk .real _ at read
  change S.lower (sharedOps (wordOps isa)) literal
    ((KernelPacket.reference model (dotModel b) b (input m) _ _).read .real)=_ at h
  rw [read] at h
  change S.lower (sharedOps (wordOps isa)) literal
      (model.cast (some .fp32) .real .real (KernelForward.values model (dotModel b) a.shape a.pid (input m) a.inputState a.shape.chunks i))=
    (wordOps isa).cast (VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history
      (sharedOps (wordOps isa)) (KernelOutput.shape a.shape)
      (sharedInputs (KernelNative.data (wordOps isa) a.shape m a.inputState Bool.false))
      a.pid.z a.shape.chunks (a.pid.x*a.shape.BK+i.1.val) (a.pid.y*a.shape.BV+i.2.1.val)) at h
  rw [shared_history] at h
  have same := history_congr (wordOps isa) a.shape
    (KernelNative.data (wordOps isa) a.shape m a.inputState Bool.false)
    (readInputs (wordOps isa).zero a m) a.pid.z rfl rfl rfl a.shape.chunks
  rw [same] at h
  simpa only [dotModel,stage,b,Reference.finalState] using h

private theorem forward_output_scalar (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call)
    (stage : a.stage=.forwardOutput) (m : RegionName→Nat→Word)
    (i : TileIndex [64,a.shape.BV])
    (hi : VeriTile.Bench.Repairs.ChunkLinearAttn.outputActive (KernelReal.parameters a.shape) a.pid i) :
    S.lower (sharedOps (wordOps isa)) literal
      (KernelOutput.outputValue model (dotModel a) a.shape a.scale a.pid (input m) i)=
    Body.output (wordOps isa) a.shape (readInputs (wordOps isa).zero a m) (literal a.scale)
      a.pid.z a.pid.y (readHistory a m Bool.false) i.1.val (a.pid.x*a.shape.BV+i.2.1.val) := by
  let b : Call:=⟨a.shape,.forwardOutput,a.scale,Bool.false,Bool.false,a.pid⟩
  have h := KernelNative.output_target isa literal laws a.shape a.scale a.pid m i hi
  have read := reference_read model (dotModel b) b (input m) ⟨2,by simp [outputs]⟩ i ⟨rfl,hi⟩
  change KernelPacket.reference model (dotModel b) b (input m) "o"
    (VeriTile.Bench.Repairs.ChunkLinearAttn.outputOffset (KernelReal.parameters a.shape) a.pid i)=Cell.mk .real _ at read
  change S.lower (sharedOps (wordOps isa)) literal
    ((KernelPacket.reference model (dotModel b) b (input m) _ _).read .real)=_ at h
  rw [read] at h
  change S.lower (sharedOps (wordOps isa)) literal
      (KernelOutput.outputValue model (dotModel b) a.shape a.scale a.pid (input m) i)=
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.output (sharedOps (wordOps isa)) (KernelOutput.shape a.shape)
      (sharedInputs (KernelNative.data (wordOps isa) a.shape m Bool.false Bool.false)) (literal a.scale)
      a.pid.z a.pid.y (readHistory a m Bool.false) i.1.val (a.pid.x*a.shape.BV+i.2.1.val) at h
  rw [shared_output] at h
  simpa only [dotModel,stage,b] using h

private theorem backward_scalar (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call)
    (stage : a.stage=.backwardState) (m : RegionName→Nat→Word) (n : Nat)
    (i : TileIndex [a.shape.BK,a.shape.BV]) :
    S.lower (sharedOps (wordOps isa)) literal
      (model.cast (some .fp32) .real .real (KernelBackward.values model (dotModel a)
        (KernelReal.parameters a.shape) a.pid (input m) a.scale a.inputState n i))=
      (wordOps isa).cast (Reference.reverseCarry (wordOps isa) a.shape
        (readInputs (wordOps isa).zero a m) (literal a.scale) a.pid.z n
        (a.pid.x*a.shape.BK+i.1.val) (a.pid.y*a.shape.BV+i.2.1.val)) := by
  have h:=KernelNative.backward_values isa literal laws a.shape a.pid m a.inputState a.scale n i
  have same:=reverse_congr (wordOps isa) a.shape
    (data (wordOps isa) a.shape m Bool.false a.inputState)
    (readInputs (wordOps isa).zero a m) (literal a.scale) a.pid.z rfl rfl rfl n
  rw [same] at h
  change (wordOps isa).cast (S.lower (sharedOps (wordOps isa)) literal
    (KernelBackward.values model (dotModel a) (KernelReal.parameters a.shape) a.pid
      (input m) a.scale a.inputState n i))=_
  apply congrArg (wordOps isa).cast
  simpa only [dotModel,stage] using h

/-- The native scalar output formulas cover every old arbitrary PID. dV's
same-program scatter uniqueness is proved separately; no host grid bounds are
introduced to use the older cross-program collision theorem. -/
theorem native_scalar_values (isa : ISA) (literal : ℝ→Word)
    (laws : S.LiteralLaws (sharedOps (wordOps isa)) literal) (a : Call) (m : RegionName→Nat→Word)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a port j) :
    mapValue (S.lower (sharedOps (wordOps isa)) literal) (outputs a.shape)[port].dtype
        (sourceValues model (dotModel a) a (input m) port j)=
      scalarOutputs (wordOps isa) (literal a.scale) a (readInputs (wordOps isa).zero a m)
        (readHistory a m Bool.false) (readHistory a m Bool.true) port j := by
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  · exact forward_history_scalar isa literal laws a active.1 m j.1.val j.1.isLt j.2 active.2
  · exact forward_final_scalar isa literal laws a active.1 m j active.2.2
  · exact forward_output_scalar isa literal laws a active.1 m j active.2
  · exact backward_scalar isa literal laws a active.1 m j.1.val j.2
  · exact backward_scalar isa literal laws a active.1 m a.shape.chunks j
  · have h := KernelNative.query_result_eq (wordOps isa) literal laws a.shape a.pid m a.scale j
    simpa only [sourceValues,scalarOutputs,outputs,mapValue,dotModel,active.1] using h
  · have h := KernelNative.key_result_eq (wordOps isa) literal laws a.shape a.pid m a.scale j active.2
    simpa only [sourceValues,scalarOutputs,outputs,mapValue,dotModel,active.1] using h
  · have h := KernelNative.value_result_eq (wordOps isa) literal laws a.shape a.pid m a.scale j.1.val j.2
    simpa only [sourceValues,scalarOutputs,outputs,mapValue,dotModel,active.1] using h

#axiomsClean native_scalar_values
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
