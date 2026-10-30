import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ValueDTypeBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    reference = tl.load(X + offsets)
    accumulator = tl.zeros((B,), dtype=tl.float32)
    for i in range(2):
        accumulator += reference.to(reference.dtype)
    tl.store(Y + offsets, accumulator)
  }

def expanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  reference = tl.load(X + offsets)
  accumulator = tl.zeros([$(B)], dtype=tl.float32)
  tl.for i in 2 {
    accumulator += reference.to(tl.float32)
  }
  tl.store(Y + offsets, accumulator)
}

theorem retained_explicit_cast (X Y : RegionName) (B : Nat) :
    source X Y B = expanded X Y B := rfl

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def changed_target (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    reference = tl.load(X + offsets)
    accumulator = tl.zeros((B,), dtype=tl.float32)
    for i in range(2):
        reference = reference.to(tl.float16)
        accumulator = accumulator.to(reference.dtype)
    tl.store(Y + offsets, accumulator)
  }


/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def narrower_target (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    reference = tl.load(X + offsets).to(tl.float16)
    accumulator = tl.zeros((B,), dtype=tl.float32)
    for i in range(2):
        accumulator = accumulator.to(reference.dtype)
    tl.store(Y + offsets, accumulator)
  }

/-- Inferring a source `.dtype` retains its explicit cast and avoids adding
an output cast that is absent from the Python arithmetic/store body. -/
def arithmeticSource (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    reference = tl.load(X + offsets)
    x = tl.load(X + offsets).to(reference.dtype)
    tl.store(Y + offsets, x * reference - x * reference)
  }

def arithmeticExpanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  reference = tl.load(X + offsets)
  x = tl.load(X + offsets).to(tl.float32)
  tl.store(Y + offsets, x * reference - x * reference)
}

theorem no_fabricated_output_cast (X Y : RegionName) (B : Nat) :
    arithmeticSource X Y B = arithmeticExpanded X Y B := rfl

#axiomsClean retained_explicit_cast
#axiomsClean no_fabricated_output_cast

end VeriTile.Tests.ValueDTypeBody
