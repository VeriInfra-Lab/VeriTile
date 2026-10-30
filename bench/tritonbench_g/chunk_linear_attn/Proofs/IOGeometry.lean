import bench.tritonbench_g.chunk_linear_attn.Proofs.IOArray
import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOGeometry
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} (c : KernelInterface.Config) (vals : ChannelValues α (IOLayout.inputs c.dims))
    (bounds : RegionName → Nat) (backed : (IOLayout.realMemory α c).InBounds bounds vals)
include backed

theorem input_capacity (i : Fin (IOLayout.inputs c.dims).length)
    (enabled : (i.val=4 → c.seed=Bool.true) ∧ (i.val=5 → c.terminal=Bool.true)) :
    TileShape.product (IOLayout.inputs c.dims)[i].shape ≤ bounds (IOLayout.inputs c.dims)[i].buf :=
  IOArray.capacity _ _ (fun j => backed.2.1 i j enabled)

theorem output_capacity (o : Fin (IOLayout.outputs c.dims).length)
    (enabled : (o.val=4 → c.final=Bool.true) ∧ (o.val=5 → c.seed=Bool.true)) :
    TileShape.product (IOLayout.outputs c.dims)[o].shape ≤ bounds (IOLayout.outputs c.dims)[o].buf :=
  IOArray.capacity _ _ (fun j => backed.2.2.1 o j enabled)

theorem scratch_capacity (k : Fin (IOLayout.scratch c.dims).length) :
    TileShape.product (IOLayout.scratch c.dims)[k].shape ≤ bounds (IOLayout.scratch c.dims)[k].buf :=
  IOArray.capacity _ _ (fun j => backed.2.2.2 k j trivial)

theorem forward_state_capacity : StageSafety.StateCapacity c.call.forward Bool.false c.seed c.final bounds := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · have h := input_capacity c vals bounds backed 1 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "key" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 2 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "value" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := scratch_capacity c vals bounds backed 0
    change (c.dims.B*c.dims.H)*(c.dims.NT*(c.dims.K*(c.dims.V*1))) ≤ bounds "history" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · intro seed
    have h := input_capacity c vals bounds backed 4 ⟨fun _ => seed,by simp⟩
    change (c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1)) ≤ bounds "initial" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · intro final
    have h := output_capacity c vals bounds backed 4 ⟨fun _ => final,by simp⟩
    change (c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1)) ≤ bounds "final" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h

theorem backward_state_capacity : StageSafety.StateCapacity c.call.backward Bool.true c.terminal c.seed bounds := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · have h := input_capacity c vals bounds backed 0 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "query" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 3 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "incoming" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := scratch_capacity c vals bounds backed 1
    change (c.dims.B*c.dims.H)*(c.dims.NT*(c.dims.K*(c.dims.V*1))) ≤ bounds "historyGradient" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · intro terminal
    have h := input_capacity c vals bounds backed 5 ⟨by simp,fun _ => terminal⟩
    change (c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1)) ≤ bounds "terminalGradient" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · intro seed
    have h := output_capacity c vals bounds backed 5 ⟨by simp,fun _ => seed⟩
    change (c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1)) ≤ bounds "initialGradient" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h

theorem output_stage_capacity : StageSafety.OutputCapacity c.call.forward bounds := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · have h := input_capacity c vals bounds backed 0 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "query" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 1 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "key" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 2 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "value" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := scratch_capacity c vals bounds backed 0
    change (c.dims.B*c.dims.H)*(c.dims.NT*(c.dims.K*(c.dims.V*1))) ≤ bounds "history" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := output_capacity c vals bounds backed 0 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "output" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h

theorem gradient_capacity : StageSafety.GradientCapacity c.call.backward bounds := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · have h := input_capacity c vals bounds backed 0 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "query" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 1 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "key" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 2 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "value" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := input_capacity c vals bounds backed 3 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "incoming" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := scratch_capacity c vals bounds backed 0
    change (c.dims.B*c.dims.H)*(c.dims.NT*(c.dims.K*(c.dims.V*1))) ≤ bounds "history" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := scratch_capacity c vals bounds backed 1
    change (c.dims.B*c.dims.H)*(c.dims.NT*(c.dims.K*(c.dims.V*1))) ≤ bounds "historyGradient" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := output_capacity c vals bounds backed 1 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "queryGradient" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := output_capacity c vals bounds backed 2 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.K*1)) ≤ bounds "keyGradient" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h
  · have h := scratch_capacity c vals bounds backed 2
    change keyBlocks (c.dims.parameters Bool.true)*((c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1))) ≤ bounds "valuePartials" at h
    simpa only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,seedName,finalName,Nat.mul_assoc,Nat.mul_one] using h

#axiomsClean forward_state_capacity
#axiomsClean backward_state_capacity
#axiomsClean output_stage_capacity
#axiomsClean gradient_capacity
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOGeometry
