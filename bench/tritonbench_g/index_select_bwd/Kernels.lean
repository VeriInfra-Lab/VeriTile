import VeriTile.Triton.DSL

/-! Separate FP32 repair. The original source and its overwrite specifications
remain unchanged. A single launch is one <=512-column slice of one index;
Python iterates index positions first, then column slices. -/
namespace VeriTile.Bench.Repairs.IndexSelectBwd
open VeriTile.Triton

structure Launch where
  (destination incoming : RegionName)
  (count stride block : Nat)

def kernel (p : Launch) : ComputeKernel :=
  let grad_source_ptr := p.destination
  let grad_output_ptr := p.incoming
  triton {
    tl.region grad_source_ptr = tl.float32, grad_output_ptr = tl.float32
    lane = tl.arange(0, $(p.block)).to(tl.int64)
    active = lane < $((p.count : Int))
    safe = tl.where(active, lane, $((0 : Int)))
    current = tl.load(grad_source_ptr + safe * $((p.stride : Int)), mask=active, other=0).to(tl.float32)
    incoming = tl.load(grad_output_ptr + safe, mask=active, other=0).to(tl.float32)
    result = current + incoming
    tl.store(grad_source_ptr + safe * $((p.stride : Int)), result, mask=active)
  }

/-- Smallest supported power-of-two lane block covering a <=512-column
slice. The original uses 512 even for a short final slice. -/
def tailBlock (count : Nat) : Nat :=
  if count ≤ 1 then 1 else if count ≤ 2 then 2 else if count ≤ 4 then 4
  else if count ≤ 8 then 8 else if count ≤ 16 then 16 else if count ≤ 32 then 32
  else if count ≤ 64 then 64 else if count ≤ 128 then 128
  else if count ≤ 256 then 256 else 512

end VeriTile.Bench.Repairs.IndexSelectBwd
