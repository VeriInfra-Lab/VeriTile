import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton

/-- The actual kernel copies bytes, for both FP32 input caches.
Rows are prepared signed int64; every index arithmetic operation is uint64.
Native range obligations are proved separately, not erased into assumptions. -/
structure Parameters where
  (cache rows output : RegionName)
  (count rowBytes start block : Nat)

/-- Source ABI: the byte views and signed row table come from the caller.
All four size/offset arguments are constexpr in the selected Python JIT. -/
def callSignature : CallSignature where
  pointers := [("Cache",.u8),("Rows",.i64),("Output",.u8)]
  scalars := []
  constexpr := ["COUNT","ROW_BYTES","START","BLOCK"]

/-- The selected Python JIT body, with its names and casts unchanged. -/
def body (Cache Rows Output : RegionName) (COUNT ROW_BYTES START BLOCK : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    idx = tl.program_id(0).to(tl.uint64) * BLOCK + tl.arange(0, BLOCK).to(tl.uint64) + START
    active = idx < COUNT
    safe = tl.where(active, idx, 0)
    row = tl.load(Rows + safe // ROW_BYTES).to(tl.uint64)
    offset = row * ROW_BYTES + safe % ROW_BYTES
    value = tl.load(Cache + offset, mask=active, other=0)
    tl.store(Output + safe, value, mask=active)
  }

/-- Strength-reduction candidate: reuse the quotient for the byte remainder.
The selected production wrapper continues to call the original Python JIT. -/
def quotientRemainderBody (Cache Rows Output : RegionName)
    (COUNT ROW_BYTES START BLOCK : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    idx = tl.program_id(0).to(tl.uint64) * BLOCK + tl.arange(0, BLOCK).to(tl.uint64) + START
    active = idx < COUNT
    safe = tl.where(active, idx, 0)
    row = tl.load(Rows + safe // ROW_BYTES).to(tl.uint64)
    offset = row * ROW_BYTES + (safe - (safe // ROW_BYTES) * ROW_BYTES)
    value = tl.load(Cache + offset, mask=active, other=0)
    tl.store(Output + safe, value, mask=active)
  }

def source (p : Parameters) : ComputeKernel :=
  body p.cache p.rows p.output p.count p.rowBytes p.start p.block

def quotientRemainder (p : Parameters) : ComputeKernel :=
  quotientRemainderBody p.cache p.rows p.output p.count p.rowBytes p.start p.block

inductive Storage where
  | fp32
  deriving DecidableEq, Repr

def Storage.bytes : Storage → Nat
  | .fp32 => 4

/-- The selected source precision is FP32 for both input caches. -/
def defaultStorage : Storage := .fp32

end VeriTile.Bench.Repairs.CacheTransform
