import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.KernelBytes
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.IntScaledMatmul.KernelNative
open VeriTile.Triton FP.Structural
open Revision001 Contract TensorBinding InputAdapter FiniteExecution FiniteIndex
noncomputable section

/-- Compatibility name for the shared exact all-word multiplication law. -/
theorem multiply_comm (cfg : FP.Config) (a b : Word) : FP.mul cfg a b=FP.mul cfg b a :=
  FP.ScalarOpsLaws.mul_comm cfg a b

/-- The inspected scaled PTX multiplies row scale by converted accumulator.
Keep that operand order explicitly in the native instruction profile. -/
def instructions (undefined : Word → BitVec 32) : CheckedNative.Instructions where
  promote32 := fun z => FP.round .fp32 Native.sourceConfig (z : Rat)
  multiply32 := fun converted rowScale => FP.mul Native.sourceConfig rowScale converted
  signed32 := KernelModel.signed32 undefined

theorem instructions_eq (undefined : Word → BitVec 32) :
    instructions undefined=KernelModel.instructions undefined := by
  unfold instructions KernelModel.instructions
  congr 1
  funext converted rowScale
  exact multiply_comm Native.sourceConfig rowScale converted

def ports (undefined : Word → BitVec 32) := CheckedNative.ports (instructions undefined)
theorem ports_eq (undefined : Word → BitVec 32) : ports undefined=KernelModel.ports undefined := by
  simp only [ports,instructions_eq,KernelModel.ports]

def run (scaled : Bool) (undefined : Word → BitVec 32)
    (a b : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) :=
  if Empty a.metadata b.metadata then some s
  else if scaled then CheckedNative.grid (ports undefined) (launch a b p) scales s
    else IntegerInstructions.intGrid (launch a b p) s

def bytesRun (scaled : Bool) (undefined : Word → BitVec 32) (a b c : TensorBinding.Tensor)
    (p : Contract.Config) (initial : ByteExecution.Machine) : Option ByteExecution.Machine :=
  if Empty a.metadata b.metadata then some initial
  else ByteExecution.sequence
    (ByteExecution.one (launch a b p) c scaled (fun s =>
      if scaled then CheckedNative.suffix (ports undefined) (launch a b p) scales s
      else fun carry i => carry i))
    (List.range (count (rows a.metadata) (columns b.metadata) p)) initial

/-- Lift the observed scalar ordering through the complete instruction grid. -/
theorem run_eq (scaled : Bool) (undefined : Word → BitVec 32)
    (a b : TensorBinding.Tensor) (p : Contract.Config) (initial : State Word) :
    run scaled undefined a b p initial=KernelExecution.native scaled undefined a b p initial := by
  simp only [run,ports_eq,KernelExecution.native]

/-- The same ordering refinement reaches the actual physical store executor. -/
theorem bytesRun_eq (scaled : Bool) (undefined : Word → BitVec 32)
    (a b c : TensorBinding.Tensor) (p : Contract.Config) (initial : ByteExecution.Machine) :
    bytesRun scaled undefined a b c p initial=KernelBytes.run scaled undefined a b c p initial := by
  simp only [bytesRun,ports_eq,KernelBytes.run,KernelBytes.suffix]
  rfl

#axiomsClean multiply_comm
#axiomsClean instructions_eq
#axiomsClean run_eq
#axiomsClean bytesRun_eq
end
end VeriTile.Bench.Repairs.IntScaledMatmul.KernelNative
