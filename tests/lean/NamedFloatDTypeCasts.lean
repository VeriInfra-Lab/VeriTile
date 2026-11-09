import tests.lean.NamedFloatDTypeCastReference
import VeriTile.Triton.DSL.NamedFloatDTypeCasts
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.NamedFloatDTypeCasts
open VeriTile.Triton

def signature : CallSignature :=
  {NamedFloatDTypeCastReference.signature with constexpr := ["B","dot_dtype"]}

def source (X Y : RegionName) (N : Int) (B : Nat) (dot_dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.NamedFloatDTypeCasts.signature call_args [N := N] in triton {
    for row in range(0, tl.cdiv(N, B)):
      value = tl.load(X, mask=row < N, other=0.0).to(dot_dtype)
      tl.store(Y, value)
  }

/-- Full-program equality for all six dtypes, signed N and every block value;
it checks the loop, mask, original cast and implicit destination conversion. -/
theorem complete_program : source=NamedFloatDTypeCastReference.reference := by
  funext X Y N B dot_dtype
  cases dot_dtype <;> rfl

/-- error: FloatDType constexpr parameters must remain immutable -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dot_dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.NamedFloatDTypeCasts.signature call_args [N := N] in triton {
    dot_dtype = tl.float16
    value = tl.load(X).to(dot_dtype)
    tl.store(Y, value)
  }

/-- error: a loop cannot shadow a FloatDType constexpr argument -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dot_dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.NamedFloatDTypeCasts.signature call_args [N := N] in triton {
    for dot_dtype in range(0, B):
      value = tl.load(X).to(dot_dtype)
      tl.store(Y, value)
  }

/-- error: a FloatDType constexpr may occur only in a supported cast dtype position -/
#guard_msgs (error) in
example (X Y : RegionName) (N : Int) (B : Nat) (dot_dtype : FloatDType) : ComputeKernel :=
  with_triton_call VeriTile.Tests.NamedFloatDTypeCasts.signature call_args [N := N] in triton {
    value = dot_dtype
    tl.store(Y, value)
  }

#axiomsClean complete_program
#auditModuleAxioms
end VeriTile.Tests.NamedFloatDTypeCasts
