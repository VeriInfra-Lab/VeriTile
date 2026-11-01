import tests.lean.FloatDTypeConstantReference
import VeriTile.Triton.DSL.FloatDTypeConstants
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural
namespace VeriTile.Tests.FloatDTypeConstants
open VeriTile.Triton FloatDTypeConstantReference

def signature : CallSignature :=
  {FloatDTypeConstantReference.signature with constexpr := ["B","dtype"]}

def source (X Y : RegionName) (N : Int) (B : Nat) (dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.FloatDTypeConstants.signature call_args [N := N] in triton {
    zero = tl.full((), 0, tl.int64)
    group = tl.full((), B, tl.int64)
    one = tl.full((), 1, tl.int64)
    last = N + group
    for row in range(N, last, one):
      value = tl.load(X)
      tl.store(Y, value.to(dtype))
  }

/-- Every floating dtype and signed starting row retains exactly the same
complete program, including the cast, both constructors and signed loop. -/
theorem complete_program : source=FloatDTypeConstantReference.reference := by
  funext X Y N B dtype
  cases dtype <;> rfl

/-- error: FloatDType constexpr parameters must remain immutable -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.FloatDTypeConstants.signature call_args [N := N] in triton {
    dtype = tl.float16
    value = tl.load(X)
    tl.store(Y, value.to(dtype))
  }

/-- error: a FloatDType constexpr may occur only in a supported cast dtype position -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.FloatDTypeConstants.signature call_args [N := N] in triton {
    value = dtype
    tl.store(Y, value)
  }

/-- error: a constant signed stride must remain immutable -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.FloatDTypeConstants.signature call_args [N := N] in triton {
    one = tl.full((), 1, tl.int64)
    one += 1
    for row in range(N, N + B, one):
      value = tl.load(X)
      tl.store(Y, value.to(dtype))
  }

/-- error: signed stride literal must be positive and fit int64 -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.FloatDTypeConstants.signature call_args [N := N] in triton {
    one = tl.full((), 0, tl.int64)
    for row in range(N, N + B, one):
      value = tl.load(X)
      tl.store(Y, value.to(dtype))
  }

/-- Constant propagation uses the value of the retained int64 constructor;
it preserves the signed start and finish and the actual signed-loop body. -/
theorem signed_stride_value {A : Type} [Inhabited A]
    (M : FP.Structural.Algebra A) (s : FP.Structural.State A)
    (h : s.regs .int [] "one" = some (fun _ => (1 : Int))) :
    FP.Structural.evalOp M none (Op.castIntToNat (Op.ref .int [] "one")) s =
      FP.Structural.evalOp M none (Op.castIntToNat (Op.castNatToInt (Op.constNat 1))) s := by
  simp [FP.Structural.evalOp_unfold,h]

#axiomsClean complete_program
#axiomsClean signed_stride_value
end VeriTile.Tests.FloatDTypeConstants
