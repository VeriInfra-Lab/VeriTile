import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.RawPythonTuplesAndStores
open VeriTile.Triton

def byteSignature : CallSignature where
  pointers := [("X",.u8),("Y",.u8)]
  scalars := []
  constexpr := ["B","N"]

def positional (X Y : RegionName) (B N : Nat) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    offsets = tl.arange(0, B)
    value = tl.load(X + offsets)
    tl.store(Y + offsets, value, offsets < N)
  }

/-- Positional masks retain the same destination narrowing and active lanes
as keyword masks. The Nat carrier does not eliminate uint8 conversion. -/
theorem positional_store (X Y : RegionName) (B N : Nat) : positional X Y B N = (triton {
    tl.region X = tl.uint8, Y = tl.uint8
    offsets = tl.arange(0, B)
    value = tl.load(X + offsets)
    tl.store(Y + offsets, value.to(tl.uint8), mask=offsets < N)
  }) := rfl

def tupleSignature : CallSignature where
  pointers := [("Y",.u8)]
  scalars := [("N",.i32)]

def tupleBody (Y : RegionName) (n : Nat) : ComputeKernel :=
  with_triton_call tupleSignature call_args [N := n] in triton {
    a, b = (N + 1, N + 2)
    tl.store(Y, a + b)
  }

theorem parenthesized_tuple (Y : RegionName) (n : Nat) : tupleBody Y n = (triton {
    tl.region Y = tl.uint8
    a, b = $((n : Nat)) + $((1 : Nat)), $((n : Nat)) + $((2 : Nat))
    tl.store(Y, (a + b).to(tl.uint8))
  }) := rfl

/-- error: simultaneous tuple reads require an explicit register-capture lowering -/
#guard_msgs in
def unsupportedSwap (Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.u8)] } : CallSignature) in triton {
    a = 1
    b = 2
    a, b = (b, a)
    tl.store(Y, a + b)
  }

def maxSignature : CallSignature where
  pointers := [("X",.fp32),("V",.fp32),("I",.i64)]
  scalars := []
  constexpr := ["B"]

def maxBody (X V I : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature maxSignature in triton {
    values = tl.load(X + tl.arange(0, B))
    vmax, imax = tl.max(values, axis=0, return_indices=True)
    tl.store(V, vmax)
    tl.store(I, imax)
  }

theorem max_returns_indices (X V I : RegionName) (B : Nat) : maxBody X V I B = (triton {
    tl.region X = tl.float32, V = tl.float32, I = tl.int64
    values = tl.load(X + tl.arange(0, B))
    vmax, imax = tl.max(values, axis=0, return_indices=True)
    tl.store(V, vmax)
    tl.store(I, imax.to(tl.int64))
  }) := rfl

/-- error: tl.max tuple binding requires `return_indices=True` -/
#guard_msgs in
def wrongMaxFlag (X V I : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature maxSignature in triton {
    values = tl.load(X + tl.arange(0, B))
    vmax, imax = tl.max(values, axis=0, return_indices=False)
    tl.store(V, vmax)
    tl.store(I, imax)
  }

def staticSignature : CallSignature where
  pointers := [("Y",.u8)]
  scalars := []
  constexpr := ["N"]

def staticRange (Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature staticSignature in triton {
    for i in range(N):
        tl.store(Y + i, i)
  }

/-- Only the invocation's immutable Nat constexpr becomes a fixed-bound loop. -/
theorem static_range_constructor (Y : RegionName) (N : Nat) :
    (staticRange Y N).surfaceBody.head?.map (fun stmt => match stmt with
      | .forLoop _ count _ => count
      | _ => N+1) = some N := rfl

def dynamicSignature : CallSignature where
  pointers := [("Y",.u8)]
  scalars := [("N",.integer)]
  constexpr := []

def dynamicRange (Y : RegionName) : ComputeKernel :=
  with_triton_signature dynamicSignature in triton {
    for i in range(N):
        tl.store(Y + i, i)
  }

/-- A scalar register remains a dynamic signed range, never a static size. -/
theorem runtime_range_constructor (Y : RegionName) :
    (dynamicRange Y).surfaceBody.head?.map (fun stmt => match stmt with
      | .forRangeInt _ _ (.ref .int [] "N") _ _ => Bool.true
      | _ => Bool.false) = some Bool.true := rfl

#axiomsClean static_range_constructor
#axiomsClean runtime_range_constructor

#axiomsClean positional_store
#axiomsClean parenthesized_tuple
#axiomsClean max_returns_indices
end VeriTile.Tests.RawPythonTuplesAndStores
