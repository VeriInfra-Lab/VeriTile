import bench.repairs.tritonbench_g.matmul_triton1.Proofs.Backing

namespace VeriTile.Bench.Repairs.MatmulTriton1.Kernel
open VeriTile.Triton FP.Structural
open Revision001 SharedContract TensorBinding
noncomputable section

def outputDescriptor (x : Launch) : Host.Tensor :=
  { InputAdapter.tensor x.c x.rc with rowStride := columns x.b.metadata }

theorem parameters_adapter (x : Launch) : x.parameters =
    Host.parameters (InputAdapter.arguments x.a x.b x.ra x.rb x.defaultFP32) (outputDescriptor x) := rfl

def realTarget (x : Launch) (read : Memory Word) (decode : Word → ℝ) (i : Index x) : ℝ :=
  LogicalInputs.realExpected x.a x.b x.la x.lb read decode i.1.val i.2.val

def precisionTarget {α : Type} [Inhabited α] (P : Core.Primitives α) (x : Launch)
    (read : Memory Word) (decode : Word → α) (i : Index x) : α :=
  LogicalInputs.precisionReference P x.a x.b x.la x.lb read decode 32 i.1.val i.2.val

theorem real_reference (x : Launch) (read : Memory Word) (decode : Word → ℝ)
    (s : State ℝ) (bound : Binding x read decode s) (i : Index x) :
    Core.logical Core.realPrimitives x.parameters s i.1.val i.2.val = realTarget x read decode i := by
  rw [Core.logical_realMatmul _ _ _ _ (by norm_num [Launch.parameters])]
  exact LogicalInputs.real_reference x.a x.b x.la x.lb x.ra x.rb x.defaultFP32
    x.admitted read decode s bound.inputs (outputDescriptor x) i

theorem precision_reference {α : Type} [Inhabited α] (P : Core.Primitives α) (x : Launch)
    (read : Memory Word) (decode : Word → α) (s : State α)
    (bound : Binding x read decode s) (i : Index x) :
    Core.logical P x.parameters s i.1.val i.2.val = precisionTarget P x read decode i :=
  LogicalInputs.precision_reference P x.a x.b x.la x.lb x.ra x.rb x.defaultFP32
    x.admitted read decode s bound.inputs (outputDescriptor x) i

#specNonCircular realTarget avoiding [grid,FiniteSource.grid,image,FP.Structural.exec]
#specNonCircular precisionTarget avoiding [grid,FiniteSource.grid,image,FP.Structural.exec]
#axiomsClean real_reference
#axiomsClean precision_reference
end
end VeriTile.Bench.Repairs.MatmulTriton1.Kernel
