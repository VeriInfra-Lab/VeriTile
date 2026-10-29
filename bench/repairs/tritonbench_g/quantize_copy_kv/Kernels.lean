import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelSource
open Lean VeriTile.Triton

structure Parameters where
  heads : Nat
  width : Nat
  height : Nat

/-! The selected contiguous FP32 source, with FP16 scale storage and signed
int8 output. Runtime strides are the exact values supplied by the canonical
Python wrapper. The candidate changes only one integer address product.
No floating arithmetic, reduction order, cast, mask or input guard changes. -/

/-- Both complete original and candidate quotations are written directly.
The explicit ports retain the existing typed IO metadata exactly. -/
def kernel (version : Bool) : Parameters → ComputeKernel :=
  match version with
  | Bool.false => fun p =>
    let K : RegionName := "input"
    let DestLoc : Region .int := ⟨"destination"⟩
    let Out : RegionName := "output"
    let OutScale : RegionName := "scale"
    triton {
      tl.ports inputs=[K, DestLoc.cast], outputs=[Out, OutScale]
      tl.region K = tl.float32, DestLoc = tl.int64, Out = tl.int8, OutScale = tl.float16
      cur_index = tl.program_id(0).to(tl.int64)
      offs_h = tl.arange(0, $(p.height)).to(tl.int64)
      offs_d = tl.arange(0, $(p.width)).to(tl.int64)
      dest_index = tl.load(DestLoc + cur_index * $(1))
      src_data = tl.load(
        K + cur_index * $(p.heads * p.width) + offs_h[:, None] * $(p.width)
          + $(1) * offs_d[None, :],
        mask=offs_h[:, None] < $(p.heads), other=0.0)
      abs_data = tl.abs(src_data)
      maximum = tl.max(abs_data, axis=1)
      data_scale = (maximum / 127.0).to(tl.float16)[:, None]
      denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
      q_src_data = (src_data / denominator).to(tl.int8)
      o_ptrs = Out + dest_index * $(p.heads * p.width) + $(p.width) * offs_h[:, None] + $(1) * offs_d[None, :]
      os_ptrs = OutScale + dest_index * $(p.heads) + $(1) * offs_h[:, None]
      tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(p.heads))
      tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
    }
  | Bool.true => fun p =>
    let K : RegionName := "input"
    let DestLoc : Region .int := ⟨"destination"⟩
    let Out : RegionName := "output"
    let OutScale : RegionName := "scale"
    triton {
      tl.ports inputs=[K, DestLoc.cast], outputs=[Out, OutScale]
      tl.region K = tl.float32, DestLoc = tl.int64, Out = tl.int8, OutScale = tl.float16
      cur_index = tl.program_id(0).to(tl.int64)
      offs_h = tl.arange(0, $(p.height)).to(tl.int64)
      offs_d = tl.arange(0, $(p.width)).to(tl.int64)
      dest_index = tl.load(DestLoc + cur_index * $(1))
      src_data = tl.load(
        K + cur_index * $(p.heads * p.width) + offs_h[:, None] * $(p.width)
          + offs_d[None, :] * $(1),
        mask=offs_h[:, None] < $(p.heads), other=0.0)
      abs_data = tl.abs(src_data)
      maximum = tl.max(abs_data, axis=1)
      data_scale = (maximum / 127.0).to(tl.float16)[:, None]
      denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
      q_src_data = (src_data / denominator).to(tl.int8)
      o_ptrs = Out + dest_index * $(p.heads * p.width) + $(p.width) * offs_h[:, None] + $(1) * offs_d[None, :]
      os_ptrs = OutScale + dest_index * $(p.heads) + $(1) * offs_h[:, None]
      tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(p.heads))
      tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
    }

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelSource
