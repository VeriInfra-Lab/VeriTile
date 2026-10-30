import bench.repairs.tritonbench_g.attention_score.Proofs.Contract
import bench.repairs.tritonbench_g.attention_score.Proofs.CurrentExecution
namespace VeriTile.Bench.Repairs.AttentionScore.Host
open VeriTile Triton Geometry Physical Whole002
noncomputable section

/-- Identity and metadata of the ordinary Tensor allocated by torch.zeros.
No output values or kernel behavior are assumed here. -/
structure Tensor where
  object : Nat
  device : Nat
  shape : List Nat
  dataPointer : Nat
  fp32 : Bool
  contiguous : Bool

structure Allocation (c : Shape) (abi : ABI c) where
  output : Tensor
  inputDevice : Nat
  shape : output.shape=[c.Z,c.H,c.NK]
  device : output.device=inputDevice
  precision : output.fp32=Bool.true
  contiguous : output.contiguous=Bool.true
  pointer : output.dataPointer=4*(abi.alloc "out").base

structure Returned (α : Type) where
  tensor : Tensor
  memory : Nat → α
  blockM : Nat
  blockN : Nat

def realReturn (c : Shape) (dispatch : Dispatch c) (abi : ABI c) (allocation : Allocation c abi)
    (physical : Nat → ℝ) (offset size : Int) (window complement : Bool) : Option (Returned ℝ) :=
  (Execution.ExactReal.execute c abi physical offset size (LogicalFormula.hostScale c.D) window complement).map
    fun memory=>⟨allocation.output,memory,persistentBlock dispatch,persistentBlock dispatch⟩

def nativeReturn (c : Shape) (dispatch : Dispatch c) (abi : ABI c) (allocation : Allocation c abi)
    (physical : Nat → BitVec 32) (offset size : Int) (hostScale : ℝ)
    (window complement : Bool) : Option (Returned NativeInstructions.Graph) :=
  (Execution.Native.execute c abi physical offset size hostScale window complement).map
    fun memory=>⟨allocation.output,memory,persistentBlock dispatch,persistentBlock dispatch⟩

#specNonCircular Allocation avoiding [realReturn,nativeReturn,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionScore.Host
