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

def source (p : Parameters) : ComputeKernel :=
  let Cache := p.cache
  let Rows := p.rows
  let Output := p.output
  triton {
    tl.ports inputs=[Cache, Rows], outputs=[Output]
    tl.region Cache = tl.uint8, Rows = tl.int64, Output = tl.uint8
    idx = tl.program_id(0).to(tl.uint64) * $(p.block) +
      tl.arange(0, $(p.block)).to(tl.uint64) + $(p.start)
    active = idx < $(p.count)
    safe = tl.where(active, idx, $(0))
    row = tl.load(Rows + safe // $(p.rowBytes)).to(tl.uint64)
    offset = row * $(p.rowBytes) + safe % $(p.rowBytes)
    value = tl.load(Cache + offset, mask=active, other=$(0))
    tl.store(Output + safe, value, mask=active)
  }

/-- Related address optimization: derive the byte remainder from the quotient
already used by the row lookup. This exposes a shared unsigned division and
replaces the separate remainder expression with multiply/subtract. The complete
launch, signed row load, uint64 casts and both masks are unchanged. -/
def quotientRemainder (p : Parameters) : ComputeKernel :=
  let Cache := p.cache
  let Rows := p.rows
  let Output := p.output
  triton {
    tl.ports inputs=[Cache, Rows], outputs=[Output]
    tl.region Cache = tl.uint8, Rows = tl.int64, Output = tl.uint8
    idx = tl.program_id(0).to(tl.uint64) * $(p.block) +
      tl.arange(0, $(p.block)).to(tl.uint64) + $(p.start)
    active = idx < $(p.count)
    safe = tl.where(active, idx, $(0))
    row = tl.load(Rows + safe // $(p.rowBytes)).to(tl.uint64)
    offset = row * $(p.rowBytes) + (safe - (safe // $(p.rowBytes)) * $(p.rowBytes))
    value = tl.load(Cache + offset, mask=active, other=$(0))
    tl.store(Output + safe, value, mask=active)
  }

inductive Storage where
  | fp32
  deriving DecidableEq, Repr

def Storage.bytes : Storage → Nat
  | .fp32 => 4

/-- The selected source precision is FP32 for both input caches. -/
def defaultStorage : Storage := .fp32

end VeriTile.Bench.Repairs.CacheTransform
