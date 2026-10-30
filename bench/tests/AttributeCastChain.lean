import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.AttributeCastChain
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Narrow",.fp16),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X","Narrow"]
  writePorts := ["Y"]

def pointerChain (X Narrow Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    for i in range(2):
        x = x.to(Narrow.dtype.element_ty).to(tl.float32)
    tl.store(Y + offsets, x)
  }

def pointerExpanded (X Narrow Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    for i in range(2):
        x = (x.to(tl.float16)).to(tl.float32)
    tl.store(Y + offsets, x)
  }

/-- The FP16 conversion remains inside every iteration, followed by FP32.
This is complete executable equality, including reads and the final store. -/
theorem pointer_roundtrip_retained (X Narrow Y : RegionName) (B : Nat) :
    pointerChain X Narrow Y B = pointerExpanded X Narrow Y B := rfl

def valueChain (X Narrow Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    reference = tl.load(Narrow + offsets)
    x = tl.load(X + offsets)
    for i in range(2):
        x = x.to(reference.dtype).to(tl.float32)
    tl.store(Y + offsets, x)
  }

def valueExpanded (X Narrow Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    reference = tl.load(Narrow + offsets)
    x = tl.load(X + offsets)
    for i in range(2):
        x = (x.to(tl.float16)).to(tl.float32)
    tl.store(Y + offsets, x)
  }

theorem value_roundtrip_retained (X Narrow Y : RegionName) (B : Nat) :
    valueChain X Narrow Y B = valueExpanded X Narrow Y B := rfl

#axiomsClean pointer_roundtrip_retained
#axiomsClean value_roundtrip_retained
end VeriTile.Tests.AttributeCastChain
