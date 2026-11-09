import VeriTile.Triton.DSL.SignedByteReads
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedByteReads
open VeriTile.Triton FP.Structural
noncomputable section
def profile : CallSignature where
  pointers := [("input",.i8)]
def copied (input : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    value = tl.load(input)
  }
theorem byte_load_exact (input : RegionName) : (copied input).surfaceBody =
    [.assign .int [] "value" (.alg (.intToSigned 8 (.load .int (.region (Region.cast input) (.constNat 0)) .none .zero)))] := by rfl
/-- Every integer memory carrier is interpreted as a signed byte, including
negative and out-of-range carriers. No numerical input premise is required. -/
theorem byte_value_exact {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (input : RegionName) :
    FP.Structural.evalOp M none (Op.intToSigned 8 (Op.load .int (.region (Region.cast input) (.constNat 0)) .none .zero)) s =
      some (fun _=> (BitVec.ofInt 8 ((s.mem input 0).read .int)).toInt) := by
  rw [FP.Structural.eval_intToSigned]
  rfl
#axiomsClean byte_load_exact
#axiomsClean byte_value_exact
def vectorProfile : CallSignature where
  pointers := [("input",.i8)]
  constexpr := ["N"]
def integerProfile : CallSignature where
  pointers := [("input",.i64)]
  constexpr := ["N"]
def maskedByte (input : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call vectorProfile call_args [] in triton {
    p = input + tl.arange(0, N)
    value = tl.load(p, mask=tl.arange(0, N)<N, other=0.0)
  }
def maskedDirect (input : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call integerProfile call_args [] in triton {
    p = input + tl.arange(0, N)
    value = tl.load(p, mask=tl.arange(0, N)<N, other=0.0).to(tl.int8)
  }
theorem pointer_mask_exact (input : RegionName) (N : Nat) : maskedByte input N=maskedDirect input N := by rfl
#axiomsClean pointer_mask_exact
/-- error: signed-byte read requires unambiguous pointer storage -/
#guard_msgs in
example (input scale : RegionName) : ComputeKernel :=
  with_triton_call {pointers := [("input",.i8),("scale",.fp32)]} call_args [] in triton {
    p = input + 0
    value = tl.load(p)
    p = scale + 0
  }
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.SignedByteReads
