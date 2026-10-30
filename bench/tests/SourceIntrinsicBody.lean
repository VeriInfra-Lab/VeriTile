import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SourceIntrinsicBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]
  unaryImports := [("tanh","triton.language.extra.libdevice.tanh")]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = tanh(x)
    tl.store(Y + offsets, y)
  }

def expanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  y = libdevice.tanh(x)
  tl.store(Y + offsets, y)
}

/-- The imported operation and store dtype are preserved without a fabricated cast. -/
theorem import_and_store (X Y : RegionName) (B : Nat) :
    source X Y B = expanded X Y B := rfl

def renamedSignature : CallSignature :=
  { signature with unaryImports := [("device_tanh","triton.language.extra.cuda.libdevice.tanh")] }

def renamed (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature renamedSignature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = device_tanh(x)
    tl.store(Y + offsets, y)
  }

theorem alias_is_import_bound (X Y : RegionName) (B : Nat) :
    renamed X Y B = expanded X Y B := rfl

def missingSignature : CallSignature := { signature with unaryImports := [] }
/-- error: bare tanh requires an explicit checked source import binding -/
#guard_msgs in
def missing (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature missingSignature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = tanh(x)
    tl.store(Y + offsets, y)
  }

def wrongSignature : CallSignature :=
  { signature with unaryImports := [("tanh","triton.language.tanh")] }
/-- error: unsupported source intrinsic import target -/
#guard_msgs in
def wrong (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature wrongSignature in triton {
    tl.store(Y, 0)
  }

/-- error: source helper and import names cannot be rebound -/
#guard_msgs in
def rebound (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature renamedSignature in triton {
    device_tanh = 1
    tl.store(Y, 0)
  }

/-- error: source intrinsic aliases may only occur in direct calls and cannot be rebound -/
#guard_msgs in
def indirect (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature renamedSignature in triton {
    f = device_tanh
    tl.store(Y, 0)
  }

#axiomsClean import_and_store
#axiomsClean alias_is_import_bound
end VeriTile.Tests.SourceIntrinsicBody
