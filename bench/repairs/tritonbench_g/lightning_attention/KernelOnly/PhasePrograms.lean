import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Algebras
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Grid
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]
def executeForward (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions) (m : Memory α) :=
  runGrid M D .forward d (grid .forward d) m
def executeBackward (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions) (m : Memory α) : Option (Memory α) := do
  let i ← runGrid M D .intra d (grid .intra d) m
  runGrid M D .inter d (grid .inter d) i
def modelForward (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions) (m : Memory α) :=
  gridMemory M D .forward d (grid .forward d) m
def modelBackward (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions) (m : Memory α) :=
  gridMemory M D .inter d (grid .inter d) (gridMemory M D .intra d (grid .intra d) m)
theorem forward_execution (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α) (d : Host.Dimensions) (m : Memory α) :
    executeForward M D d m=some (modelForward M D d m) := grid_execution M hm D _ _ _ _
theorem backward_execution (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α) (d : Host.Dimensions) (m : Memory α) :
    executeBackward M D d m=some (modelBackward M D d m) := by
  simp only [executeBackward,grid_execution M hm,bind,Option.bind_some,modelBackward]
#axiomsClean forward_execution
#axiomsClean backward_execution
#specNonCircular modelBackward avoiding [executeBackward,runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
