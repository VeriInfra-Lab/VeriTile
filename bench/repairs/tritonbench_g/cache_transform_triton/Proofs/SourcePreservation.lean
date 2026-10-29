import bench.repairs.tritonbench_g.cache_transform_triton.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only snapshots of the previous selected programs. The current
Kernels.lean is authoritative and does not import these snapshots. The equalities
include the complete port lists and statement trees, not just an observed value. -/
namespace VeriTile.Bench.Repairs.CacheTransform.SourcePreservation
open VeriTile.Triton

def previousSource (p : Parameters) : ComputeKernel :=
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
def previousCandidate (p : Parameters) : ComputeKernel :=
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


theorem original (p : Parameters) : source p = previousSource p := rfl
theorem candidate (p : Parameters) : quotientRemainder p = previousCandidate p := rfl

#axiomsClean original
#axiomsClean candidate
end VeriTile.Bench.Repairs.CacheTransform.SourcePreservation
