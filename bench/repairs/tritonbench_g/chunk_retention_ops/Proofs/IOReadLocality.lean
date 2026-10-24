import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Equality of precisely the finite active input read values. No relation is
required on unrelated memory, dormant optional inputs, or cell tags discarded
by the existing .real source read representation. -/
def ReadEq (a : Call) (m n : KernelPacket.Memory α) : Prop :=
  ∀ port j,inputEnabled a port →
    (m (inputs a.shape)[port].buf (inputOffset a port j)).read .real=
      (n (inputs a.shape)[port].buf (inputOffset a port j)).read .real

namespace FH
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
  (initValue keyLoad valueLoad gatedValue update active stateAddress feature column)
end FH
namespace FO
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
  (featureLoad keyLoad historyLoad queryGate update accumulated valueLoad value active outputAddress row column)
end FO

private theorem forward_initial (M : Algebra α) (a : Call) (stage : a.stage=.forwardState)
    (m n : KernelPacket.Memory α) (reads : ReadEq a m n) :
    FH.initValue M (KernelOutput.shape a.shape) a.inputState (KernelForward.state a.shape a.pid m)=
      FH.initValue M (KernelOutput.shape a.shape) a.inputState (KernelForward.state a.shape a.pid n) := by
  funext i
  unfold FH.initValue
  dsimp only [FH.active,FH.feature,FH.column,KernelForward.state,
    VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
  split_ifs with seed active
  · congr 1
    exact reads ⟨4,by simp [inputs]⟩ (⟨_,active.1⟩,⟨_,active.2⟩,()) ⟨stage,seed⟩
  · rfl
  · rfl

private theorem forward_key (M : Algebra α) (a : Call) (stage : a.stage=.forwardState)
    (m n : KernelPacket.Memory α) (reads : ReadEq a m n) (block : Nat) :
    FH.keyLoad M (KernelOutput.shape a.shape) (KernelForward.state a.shape a.pid m) block=
      FH.keyLoad M (KernelOutput.shape a.shape) (KernelForward.state a.shape a.pid n) block := by
  funext i
  unfold FH.keyLoad
  congr 1
  dsimp only [FH.feature,FH.column,FH.active,FO.row,FO.column,FO.active,KernelForward.state,KernelOutput.state,
    VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
  split_ifs with active
  · simpa [inputOffset,inputs,FH.feature,KernelForward.state,KernelOutput.shape,parameters,
      VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
        reads ⟨1,by simp [inputs]⟩ (⟨_,active.2⟩,⟨_,active.1⟩,()) (Or.inl stage)
  · rfl

private theorem forward_value (M : Algebra α) (a : Call) (stage : a.stage=.forwardState)
    (m n : KernelPacket.Memory α) (reads : ReadEq a m n) (block : Nat) :
    FH.valueLoad M (KernelOutput.shape a.shape) (KernelForward.state a.shape a.pid m) block=
      FH.valueLoad M (KernelOutput.shape a.shape) (KernelForward.state a.shape a.pid n) block := by
  funext i
  unfold FH.valueLoad
  congr 1
  dsimp only [FH.feature,FH.column,FH.active,FO.row,FO.column,FO.active,KernelForward.state,KernelOutput.state,
    VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
  split_ifs with active
  · exact reads ⟨2,by simp [inputs]⟩ (⟨_,active.1⟩,⟨_,active.2⟩,()) (Or.inl stage)
  · rfl

theorem forward_values_readEq (M : Algebra α) (D : DotPrimitive α) (a : Call)
    (stage : a.stage=.forwardState) (m n : KernelPacket.Memory α) (reads : ReadEq a m n) (count : Nat) :
    KernelForward.values M D a.shape a.pid m a.inputState count=
      KernelForward.values M D a.shape a.pid n a.inputState count := by
  induction count with
  | zero => exact forward_initial M a stage m n reads
  | succ count ih =>
    unfold KernelForward.values
    rw [ih]
    funext i
    unfold FH.update FH.gatedValue
    rw [forward_key M a stage m n reads,forward_value M a stage m n reads]
    rfl

/-- The output stage's finite input windows determine both block accumulators
and the final value load. -/
theorem output_value_readEq (M : Algebra α) (D : DotPrimitive α) (a : Call)
    (stage : a.stage=.forwardOutput) (m n : KernelPacket.Memory α) (reads : ReadEq a m n) :
    KernelOutput.outputValue M D a.shape a.scale a.pid m=
      KernelOutput.outputValue M D a.shape a.scale a.pid n := by
  have query (block : Nat) :
      FO.featureLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid m) "q" block=
      FO.featureLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) "q" block := by
    funext i
    unfold FO.featureLoad
    congr 1
    dsimp only [FH.feature,FH.column,FH.active,FO.row,FO.column,FO.active,KernelForward.state,KernelOutput.state,
      VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
    split_ifs with active
    · exact reads ⟨0,by simp [inputs]⟩ (⟨_,active.1⟩,⟨_,active.2⟩,()) (Or.inl stage)
    · rfl
  have key (block : Nat) :
      FO.keyLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid m) block=
      FO.keyLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) block := by
    funext i
    unfold FO.keyLoad
    congr 1
    dsimp only [FH.feature,FH.column,FH.active,FO.row,FO.column,FO.active,KernelForward.state,KernelOutput.state,
      VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
    split_ifs with active
    · simpa [inputOffset,inputs,FO.row,KernelOutput.state,KernelOutput.shape,parameters,
        VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState,
        Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
          reads ⟨1,by simp [inputs]⟩ (⟨_,active.2⟩,⟨_,active.1⟩,()) (Or.inr (Or.inl stage))
    · rfl
  have hist (block : Nat) :
      FO.historyLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid m) block=
      FO.historyLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) block := by
    funext i
    unfold FO.historyLoad
    congr 1
    dsimp only [FH.feature,FH.column,FH.active,FO.row,FO.column,FO.active,KernelForward.state,KernelOutput.state,
      VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
    split_ifs with active
    · exact reads ⟨6,by simp [inputs]⟩ (⟨_,active.1⟩,⟨_,active.2⟩,()) (Or.inl stage)
    · rfl
  have gate (block : Nat) :
      FO.queryGate M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid m) block=
      FO.queryGate M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) block := by
    funext i
    unfold FO.queryGate
    rw [query]
    rfl
  have updates : FO.update M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape)
      (KernelOutput.state a.shape a.pid m)=
      FO.update M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) := by
    funext block acc
    simp only [FO.update,gate,query,key,hist]
  have accum : FO.accumulated M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape)
      (KernelOutput.state a.shape a.pid m)=
      FO.accumulated M (KernelOutput.sharedDot D) (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) := by
    unfold FO.accumulated
    rw [updates]
  have val : FO.valueLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid m)=
      FO.valueLoad M (KernelOutput.shape a.shape) (KernelOutput.state a.shape a.pid n) := by
    funext i
    unfold FO.valueLoad
    congr 1
    dsimp only [FH.feature,FH.column,FH.active,FO.row,FO.column,FO.active,KernelForward.state,KernelOutput.state,
      VeriTile.Bench.Repairs.ChunkLinearAttn.launchState,VeriTile.Bench.Repairs.ChunkDeltaFwd.memoryState]
    split_ifs with active
    · exact reads ⟨2,by simp [inputs]⟩ (⟨_,active.1⟩,⟨_,active.2⟩,()) (Or.inr (Or.inl stage))
    · rfl
  funext i
  unfold KernelOutput.outputValue FO.value KernelOutput.S.attention
  rw [accum,val]
  rfl

#axiomsClean forward_values_readEq
#axiomsClean output_value_readEq
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
