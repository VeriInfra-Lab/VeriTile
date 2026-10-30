import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

/-! Block-pointer rebinding and advance preserve source storage types. Max
promotes FP16 to FP32, then only the true narrowing store conversion remains. -/
namespace VeriTile.Tests.BlockPointerDType
open VeriTile.Triton

def profile16 : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source16 (X Y : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature profile16 in triton {
    X = tl.make_block_ptr(base=X, shape=(M, N), strides=(N, 1),
      offsets=(0, 0), block_shape=(M, N), order=(1, 0))
    ptr = tl.advance(X, (0, 0))
    x = tl.load(ptr)
    y = tl.max(x, axis=1)
    tl.store(Y + tl.arange(0, M), y)
  }

def expanded16 (X Y : RegionName) (M N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16, Y = tl.float16
  X = tl.make_block_ptr(base=X, shape=($(M), $(N)), strides=($(N), $(1)),
    offsets=($(0), $(0)), block_shape=($(M), $(N)), order=(1, 0))
  ptr = tl.advance(X, ($(0), $(0)))
  x = tl.load(ptr)
  y = tl.max(x, axis=1)
  tl.store(Y + tl.arange(0, $(M)), y.to(tl.float16))
}

theorem complete_source16 (X Y : RegionName) (M N : Nat) :
    source16 X Y M N = expanded16 X Y M N := rfl
#axiomsClean complete_source16

def profile32 : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source32 (X Y : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature profile32 in triton {
    X = tl.make_block_ptr(base=X, shape=(M, N), strides=(N, 1),
      offsets=(0, 0), block_shape=(M, N), order=(1, 0))
    ptr = tl.advance(X, (0, 0))
    x = tl.load(ptr)
    y = tl.max(x, axis=1)
    tl.store(Y + tl.arange(0, M), y)
  }

def expanded32 (X Y : RegionName) (M N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  X = tl.make_block_ptr(base=X, shape=($(M), $(N)), strides=($(N), $(1)),
    offsets=($(0), $(0)), block_shape=($(M), $(N)), order=(1, 0))
  ptr = tl.advance(X, ($(0), $(0)))
  x = tl.load(ptr)
  y = tl.max(x, axis=1)
  tl.store(Y + tl.arange(0, $(M)), y)
}

theorem complete_source32 (X Y : RegionName) (M N : Nat) :
    source32 X Y M N = expanded32 X Y M N := rfl
#axiomsClean complete_source32

def profile64 : CallSignature where
  pointers := [("X",.fp64),("Y",.fp64)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source64 (X Y : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature profile64 in triton {
    X = tl.make_block_ptr(base=X, shape=(M, N), strides=(N, 1),
      offsets=(0, 0), block_shape=(M, N), order=(1, 0))
    ptr = tl.advance(X, (0, 0))
    x = tl.load(ptr)
    y = tl.max(x, axis=1)
    tl.store(Y + tl.arange(0, M), y)
  }

def expanded64 (X Y : RegionName) (M N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float64, Y = tl.float64
  X = tl.make_block_ptr(base=X, shape=($(M), $(N)), strides=($(N), $(1)),
    offsets=($(0), $(0)), block_shape=($(M), $(N)), order=(1, 0))
  ptr = tl.advance(X, ($(0), $(0)))
  x = tl.load(ptr)
  y = tl.max(x, axis=1)
  tl.store(Y + tl.arange(0, $(M)), y)
}

theorem complete_source64 (X Y : RegionName) (M N : Nat) :
    source64 X Y M N = expanded64 X Y M N := rfl
#axiomsClean complete_source64

end VeriTile.Tests.BlockPointerDType
