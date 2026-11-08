import VeriTile.Triton.DSL.PythonCallSyntax
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PythonCallSyntax
open VeriTile.Triton

#guard DSL.PythonCallSyntax.normalize "tl.where(a, b, c,)" == "tl.where(a, b, c)"
#guard DSL.PythonCallSyntax.normalize "tl.zeros((B,), tl.float32)" == "tl.zeros((B), tl.float32)"
#guard DSL.PythonCallSyntax.normalize "a = (value,)" == "a = (value,)"
#guard DSL.PythonCallSyntax.normalize "a[value,]" == "a[value,]"
#guard DSL.PythonCallSyntax.normalize "tl.load(X, other=\"a,)\",)" == "tl.load(X, other=\"a,)\")"
#guard DSL.PythonCallSyntax.normalize "# call(x,)\ncall(y,)" == "# call(x,)\ncall(y)"
#guard DSL.PythonCallSyntax.normalize "\"\"\"call(x,)\"\"\"\ncall(y,)" == "\"\"\"call(x,)\"\"\"\ncall(y)"

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]

def copied (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B,)
    x = tl.load(X + offsets,)
    acc = tl.zeros((B,), tl.float32)
    result = tl.where(x > 0.0, acc + x, x,)
    tl.store(Y + offsets, result,)
  }

/-- Delimiter normalization preserves the complete typed program for every
shape and both ports, including every load, operation, cast and store. -/
theorem complete_program (X Y : RegionName) (B : Nat) : copied X Y B =
    (with_triton_signature signature in triton {
      offsets = tl.arange(0, B)
      x = tl.load(X + offsets)
      acc = tl.zeros([B], tl.float32)
      result = tl.where(x > 0.0, acc + x, x)
      tl.store(Y + offsets, result)
    }) := rfl

#axiomsClean complete_program
end VeriTile.Tests.PythonCallSyntax
