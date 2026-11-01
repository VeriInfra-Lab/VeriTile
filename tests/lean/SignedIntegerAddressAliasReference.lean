import VeriTile.Triton.DSL.SingleAssignmentPointerOffsets
namespace VeriTile.Tests.SignedIntegerAddressAliasReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("start",.i64),("stride",.i64),("N",.i64)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

/-- Independently typed signed address expressions. The casts make source
integer dtypes explicit before the existing lowering, without an alias pass. -/
def scalar (X Y : RegionName) (base B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [X := base] in triton {
    offsets = tl.cast(start + stride, tl.int64)
    value = tl.load(X + offsets)
    tl.store(Y, value)
  }

def vector (X Y : RegionName) (base B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [X := base] in triton {
    columns = tl.cast(tl.arange(0, B).to(tl.int64), tl.int64)
    mask = columns < N
    offsets = tl.cast(start + stride * tl.where(mask, columns, 0), tl.int64)
    values = tl.load(X + offsets, mask=mask, other=0)
    tl.store(Y + columns, values, mask=mask)
  }
end VeriTile.Tests.SignedIntegerAddressAliasReference
