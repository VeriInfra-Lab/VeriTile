import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Correct
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.Endpoints
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.Safety

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.TensorBridge
open VeriTile.Triton FP.Structural Revision001 Model TensorBinding
noncomputable section
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
        after.mem tag i=(entry read c other).mem tag i) :=
  Endpoints.real read c h choice other


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
        after.mem tag i=(entry read c other).mem tag i) :=
  Endpoints.native ops read c h choice other

theorem empty {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (c : Arguments) (choice : Fin 8) (before : State α)
    (empty : rows c.a.metadata=0 ∨ columns c.b.metadata=0) :
    run rest P c choice before=some before :=
  Model.empty rest P c choice before empty

#axiomsClean real
#axiomsClean native
#axiomsClean empty
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.TensorBridge
