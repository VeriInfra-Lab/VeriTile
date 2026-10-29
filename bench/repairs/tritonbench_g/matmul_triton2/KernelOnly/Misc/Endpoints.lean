import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Targets
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Tensor

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Endpoints
open VeriTile.Triton FP.Structural Revision001 Model TensorBinding
noncomputable section
set_option maxHeartbeats 8000000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem real (read : Nat → ℝ) (c : Arguments) (h : KernelContract c)
    (choice : Fin 8) (other : State ℝ) :
    ∃ after,run (unused 0) Core.realPrimitives c choice (entry read c other)=some after ∧
      (∀ r col,r < rows c.a.metadata → col < columns c.b.metadata →
        project read c after (c.output.storage.address
          (c.output.physicalIndex (r*columns c.b.metadata+col)))=Targets.real read c r col) ∧
      (∀ a,a < c.output.metadata.dataPtr.toNat ∨
        c.output.metadata.dataPtr.toNat+4*(rows c.a.metadata*columns c.b.metadata) ≤ a →
        project read c after a=read a) ∧
      (∀ t,t=c.a ∨ t=c.b → ∀ a,t.storage.base ≤ a ∧ a < t.storage.base+4*t.storage.elements →
        project read c after a=read a) ∧
      (∀ tag i,tag ≠ ("C" : RegionName) ∨ rows c.a.metadata*columns c.b.metadata ≤ i →
        after.mem tag i=(entry read c other).mem tag i) := by
  obtain ⟨after,ran,values,frame⟩ := complete (unused 0) Core.realPrimitives c choice (entry read c other)
  refine ⟨after,ran,?_,exterior read c after,?_,frame⟩
  · intro r col hr hc
    have bound : r*columns c.b.metadata+col < rows c.a.metadata*columns c.b.metadata := by nlinarith
    rw [output_index c h _ bound]
    simp only [VeriTile.Torch.FP32.Storage.address,←output_pointer c h (by omega)]
    change project read c after (address c.output _) = _
    rw [projected read c after _ bound,values r col hr hc,Cell.read_same,Targets.real_value]
  · intro t selected a inside
    apply input_frame read c h after t _ a inside
    rcases selected with rfl|rfl
    · exact h.output.apartLeft
    · exact h.output.apartRight

theorem native (ops : Targets.Ops) (read : Nat → Word) (c : Arguments) (h : KernelContract c)
    (choice : Fin 8) (other : State Word) :
    ∃ after,run (unused (FP.Value.zero .fp32))
        (Native.primitives (Targets.instructions ops) (Native.family choice)) c choice (entry read c other)=some after ∧
      (∀ r col,r < rows c.a.metadata → col < columns c.b.metadata →
        VeriTile.Torch.FP32.logicalValue (project read c after) c.output c.outputLaws
          (r*columns c.b.metadata+col)=Targets.native ops read c choice r col) ∧
      (∀ a,a < c.output.metadata.dataPtr.toNat ∨
        c.output.metadata.dataPtr.toNat+4*(rows c.a.metadata*columns c.b.metadata) ≤ a →
        project read c after a=read a) ∧
      (∀ t,t=c.a ∨ t=c.b → ∀ a,t.storage.base ≤ a ∧ a < t.storage.base+4*t.storage.elements →
        project read c after a=read a) ∧
      (∀ tag i,tag ≠ ("C" : RegionName) ∨ rows c.a.metadata*columns c.b.metadata ≤ i →
        after.mem tag i=(entry read c other).mem tag i) := by
  obtain ⟨after,ran,values,frame⟩ := complete (unused (FP.Value.zero .fp32))
    (Native.primitives (Targets.instructions ops) (Native.family choice)) c choice (entry read c other)
  refine ⟨after,ran,?_,exterior read c after,?_,frame⟩
  · intro r col hr hc
    have bound : r*columns c.b.metadata+col < rows c.a.metadata*columns c.b.metadata := by nlinarith
    have numel : r*columns c.b.metadata+col < c.output.metadata.numel := by rwa [output_numel c h]
    rw [←logical_read c.output c.outputLaws h.output.contiguous h.output.noNeg
      (project read c after) _ numel,projected read c after _ bound,
      values r col hr hc,Cell.read_same,Targets.native_value]
  · intro t selected a inside
    apply input_frame read c h after t _ a inside
    rcases selected with rfl|rfl
    · exact h.output.apartLeft
    · exact h.output.apartRight

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Endpoints
