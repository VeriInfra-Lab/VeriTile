import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.BinaryImportBody
open VeriTile.Triton
noncomputable section

def powerFunction : SourceFunction := triton_function power(x, y) in triton {
  return _pow(x.to(tl.float32), y)
}
def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  functions := [powerFunction]
  binaryImports := [("_pow","triton.language.extra.cuda.libdevice.pow")]

def pasted (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = power(x, x)
    tl.store(Y, y)
  }
def direct (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with functions := [], binaryImports := []} : CallSignature) in triton {
    x = tl.load(X)
    y = tl.extra.cuda.libdevice.pow(x.to(tl.float32), x)
    tl.store(Y, y)
  }
theorem complete_expansion (X Y : RegionName) : pasted X Y = direct X Y := rfl

/-- error: unsupported binary source intrinsic import target -/
#guard_msgs in
def wrongTarget (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({signature with binaryImports := [("_pow","triton.language.math.pow")]} : CallSignature) in triton {
    x = tl.load(X)
    tl.store(Y, x)
  }
/-- error: source helper and import names cannot be rebound -/
#guard_msgs in
def rebound (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    _pow = tl.load(X)
    tl.store(Y, _pow)
  }
/-- error: source intrinsic aliases may only occur in direct calls and cannot be rebound -/
#guard_msgs in
def wrongArity (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = _pow(x)
    tl.store(Y, y)
  }
#axiomsClean complete_expansion
end
end VeriTile.Tests.BinaryImportBody
