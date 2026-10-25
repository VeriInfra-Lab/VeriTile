import bench.repairs.tritonbench_g.quantize_copy_kv.Kernels

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelSource
open Lean VeriTile.Triton
set_option maxHeartbeats 3000000
private def legacyKernel (version : Bool) : Parameters → ComputeKernel :=
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

/-- Complete executable-body identity against the previous original/candidate
programs, including the actual typed conversion and store graph. -/
theorem pasted_body_eq (version : Bool) (p : Parameters) :
    (kernel version p).surfaceBody = (legacyKernel version p).surfaceBody := by
  cases version <;> rfl

#axiomsClean pasted_body_eq
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelSource
