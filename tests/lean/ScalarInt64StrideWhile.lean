import VeriTile.Triton.DSL.PointerDTypeZeros
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ScalarInt64StrideWhile
open VeriTile.Triton
noncomputable section
def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["G"]
  writePorts := ["Y"]
def pasted (Y : RegionName) (N G : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] integer_args [] arange_args [] in triton {
    row = tl.program_id(0).to(tl.int64)
    N = tl.cast(N, tl.int64)
    seed = tl.zeros([], dtype=Y.dtype.element_ty)
    while row < N:
        tl.store(Y + row, seed)
        row += G
  }
theorem zero_precision (Y : RegionName) (N G : Nat) :
    (pasted Y N G).surfaceBody[2]? = some
      (.assign .real [] "seed" (.compute (.full [] (.alg .fp32 (.const 0))))) := rfl
def loadSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  readPorts := ["X"]
  writePorts := ["Y"]
def augmented (X Y : RegionName) : ComputeKernel :=
  with_triton_call loadSignature call_args [] integer_args [] arange_args [] in triton {
    acc = tl.zeros([], dtype=Y.dtype.element_ty)
    acc += tl.load(X)
    tl.store(Y, acc)
  }
theorem augmented_typed_load (X Y : RegionName) :
    (augmented X Y).surfaceBody[1]? = some
      (.assign .real [] "__veritile_load_0" (.compute (.load .fp32 (.region X (.constNat 0)) .none))) := rfl

def literalZero (Y : RegionName) : ComputeKernel :=
  let G : Nat := 1
  with_triton_call signature call_args [N := (0 : Nat)] integer_args [] arange_args [] in triton {
    index = 0
    tl.store(Y + index, 1.0)
  }
theorem literal_zero_control {α : Type} [Inhabited α] (m : FP.Structural.Algebra α)
    (Y : RegionName) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.run m ((literalZero Y).surfaceBody.take 1) s=some t ∧
      t.regs .int [] "index"=some (fun _ => 0) := by
  simp [literalZero,ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalExpr,FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.integralMod,FP.Structural.numeric,FP.Structural.numericLt,
    FP.Structural.numericEq,FP.Structural.natLt,FP.Structural.integralFloorDiv,FP.Structural.bop,FP.Structural.State.setReg,Option.bind]
#axiomsClean augmented_typed_load
#axiomsClean literal_zero_control
#axiomsClean zero_precision
end
end VeriTile.Tests.ScalarInt64StrideWhile
