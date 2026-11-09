import tests.lean.PythonFullConstructorReference
import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonFullConstructors
open VeriTile.Triton

def constructorSignature : CallSignature where
  pointers := []
  scalars := []
  constexpr := ["B"]

def constructors (B : Nat) : ComputeKernel :=
  with_triton_signature constructorSignature in triton {
    scalar = tl.full((), 1, tl.int64)
    ints = tl.full((B,), 3, tl.int64)
    floats = tl.full([B], 0.0, tl.float32)
  }
theorem complete_constructors : constructors=PythonFullConstructorReference.constructors := by rfl

def castSignature : CallSignature where
  pointers := [("X",.fp32)]
  scalars := [("base",.i64)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := []
def signedCasts (X : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature castSignature in triton {
    pid = tl.program_id(0).to(tl.int64) + base
    offset = pid.to(tl.int64) * B + tl.arange(0,B).to(tl.int64)
    value = tl.load(X + offset)
  }
theorem complete_signed_casts : signedCasts=PythonFullConstructorReference.signedCasts := by rfl
-- Unsupported values fail instead of being coerced away.
/-- error: tl.cast: signed integer casts are not modeled yet -/
#guard_msgs (error) in
example (B : Nat) : ComputeKernel :=
  with_triton_signature constructorSignature in triton {
    ints = tl.full((B,), -3, tl.int64)
  }
#axiomsClean complete_constructors
#axiomsClean complete_signed_casts
end VeriTile.Tests.PythonFullConstructors
