import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.RawCorrect
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Correct
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Audit
open VeriTile.Triton FP.Structural Revision001 Model
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

example {α : Type} (A : Algebra α) (P : Core.Primitives α) (x : α) :
    (Core.backend A P).fp32Load x=x := rfl

theorem real_zero_inner (read : Nat → ℝ) (c : Arguments) (r col : Nat)
    (zero : columns c.a.metadata=0) : Targets.real read c r col=0 := by
  simp [Targets.real,zero,RealSum.sum]

theorem native_zero_inner (ops : Targets.Ops) (read : Nat → Word) (c : Arguments)
    (choice : Fin 8) (r col : Nat) (zero : columns c.a.metadata=0) :
    Targets.native ops read c choice r col=FP.Value.zero .fp32 := by
  have positive := (Native.config_legal choice).2.2.1
  have div : ((Native.config choice).blockInner-1)/(Native.config choice).blockInner=0 :=
    Nat.div_eq_of_lt (by omega)
  simp [Targets.native,zero,div,Targets.accumulate]

theorem empty_physical {α : Type} [Inhabited α] (read : Nat → α) (c : Arguments) (after : State α)
    (empty : rows c.a.metadata=0 ∨ columns c.b.metadata=0) :
    TensorBinding.project read c after=read := by
  have zero : rows c.a.metadata*columns c.b.metadata=0 := by
    rcases empty with hm|hn
    · simp [hm]
    · simp [hn]
  funext a
  apply TensorBinding.exterior read c after a
  simp only [zero,Nat.mul_zero,Nat.add_zero]
  omega

#axiomsClean kernel_real
#axiomsClean kernel_native
#axiomsClean RawKernel.kernel_empty
#axiomsClean TensorBridge.real
#axiomsClean TensorBridge.native
#axiomsClean TensorBridge.empty
#axiomsClean empty_physical
#axiomsClean real_zero_inner
#axiomsClean native_zero_inner
#axiomsClean check_inputs_iff
#axiomsClean Safety.active_accesses
#axiomsClean Safety.pointer_windows
#axiomsClean Safety.finite_source
#specNonCircular Targets.real avoiding [Model.run,Core.grid,Core.logical,Core.source,FP.Structural.exec]
#specNonCircular Targets.native avoiding [Model.run,Core.grid,Core.logical,Core.source,FP.Structural.exec]
#specNonCircular KernelContract avoiding [Model.run,Core.grid,Core.source,FP.Structural.exec,Host.Allocator,Host.call]
#specNonCircular Arguments avoiding [Model.run,Core.grid,Core.source,FP.Structural.exec,Host.Allocator,Host.call]
#specNonCircular TensorBinding.project avoiding [Targets.real,Targets.native,Model.run,Core.grid]
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Audit
