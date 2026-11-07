import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton

/-- The actual kernel copies bytes, for both FP32 input caches.
Rows are prepared signed int64; every index arithmetic operation is uint64.
Native range obligations are proved separately, not erased into assumptions. -/
structure Parameters where
  (cache rows output : RegionName)
  (count rowBytes start block : Nat)

def callSignature : CallSignature where
  pointers := [("Cache",.u8),("Rows",.i64),("Output",.u8)]
  scalars := []
  constexpr := ["COUNT","ROW_BYTES","START","BLOCK"]

def bodyKernel (Cache Rows Output : RegionName) (COUNT ROW_BYTES START BLOCK : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    idx = tl.program_id(0).to(tl.uint64) * BLOCK + tl.arange(0, BLOCK).to(tl.uint64) + START
    active = idx < COUNT
    safe = tl.where(active, idx, 0)
    row = tl.load(Rows + safe // ROW_BYTES).to(tl.uint64)
    offset = row * ROW_BYTES + safe % ROW_BYTES
    value = tl.load(Cache + offset, mask=active, other=0)
    tl.store(Output + safe, value, mask=active)
  }

def source (p : Parameters) : ComputeKernel :=
  bodyKernel p.cache p.rows p.output p.count p.rowBytes p.start p.block

/-- Related address optimization: derive the byte remainder from the quotient
already used by the row lookup. This exposes a shared unsigned division and
replaces the separate remainder expression with multiply/subtract. The complete
launch, signed row load, uint64 casts and both masks are unchanged. -/
def quotientRemainderKernel (Cache Rows Output : RegionName)
    (COUNT ROW_BYTES START BLOCK : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    idx = tl.program_id(0).to(tl.uint64) * BLOCK + tl.arange(0, BLOCK).to(tl.uint64) + START
    active = idx < COUNT
    safe = tl.where(active, idx, 0)
    row = tl.load(Rows + safe // ROW_BYTES).to(tl.uint64)
    offset = row * ROW_BYTES + (safe - (safe // ROW_BYTES) * ROW_BYTES)
    value = tl.load(Cache + offset, mask=active, other=0)
    tl.store(Output + safe, value, mask=active)
  }

def quotientRemainder (p : Parameters) : ComputeKernel :=
  quotientRemainderKernel p.cache p.rows p.output p.count p.rowBytes p.start p.block

inductive Storage where
  | fp32
  deriving DecidableEq, Repr

def Storage.bytes : Storage → Nat
  | .fp32 => 4

/-- The selected source precision is FP32 for both input caches. -/
def defaultStorage : Storage := .fp32

end VeriTile.Bench.Repairs.CacheTransform
