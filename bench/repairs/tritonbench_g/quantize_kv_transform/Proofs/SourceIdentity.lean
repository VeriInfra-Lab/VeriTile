import bench.repairs.tritonbench_g.quantize_kv_transform.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelSource
open VeriTile.Triton
set_option maxHeartbeats 3000000
private def legacyKernel (p : Parameters) : ComputeKernel :=
  let K : RegionName := "input"
  let DestLoc : RegionName := "destination"
  let Out : RegionName := "output"
  let OutScale : RegionName := "scale"
  triton {
    tl.ports inputs=[K,DestLoc], outputs=[Out,OutScale]
    tl.region K = tl.float32, DestLoc = tl.int32, Out = tl.int8, OutScale = tl.float32
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, $(p.height)).to(tl.int64)
    offs_d = tl.arange(0, $(p.width)).to(tl.int64)
    dest_index = tl.load(DestLoc + cur_index * $(1)).to(tl.int64)
    src_data = tl.load(K + cur_index * $(p.heads * p.columns) +
      offs_h[:, None] * $(p.columns) + $(1) * offs_d[None, :],
      mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.columns)), other=0.0)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = ((maximum / 127.0).to(OutScale.dtype.element_ty))[:, None]
    denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    o_ptrs = Out + dest_index * $(p.heads * p.columns) +
      $(p.columns) * offs_h[:, None] + $(1) * offs_d[None, :]
    os_ptrs = OutScale + dest_index * $(p.heads) + $(1) * offs_h[:, None]
    tl.store(o_ptrs, q_src_data,
      mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.columns)))
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
  }

/-- Pure output pointer formation is scheduled before floating quantization.
This exposes instruction overlap without changing numerical operations. -/
private def legacyEarly (p : Parameters) : ComputeKernel :=
  let K : RegionName := "input"
  let DestLoc : RegionName := "destination"
  let Out : RegionName := "output"
  let OutScale : RegionName := "scale"
  triton {
    tl.ports inputs=[K,DestLoc], outputs=[Out,OutScale]
    tl.region K = tl.float32, DestLoc = tl.int32, Out = tl.int8, OutScale = tl.float32
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, $(p.height)).to(tl.int64)
    offs_d = tl.arange(0, $(p.width)).to(tl.int64)
    dest_index = tl.load(DestLoc + cur_index * $(1)).to(tl.int64)
    src_data = tl.load(K + cur_index * $(p.heads * p.columns) +
      offs_h[:, None] * $(p.columns) + $(1) * offs_d[None, :],
      mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.columns)), other=0.0)
    o_ptrs = Out + dest_index * $(p.heads * p.columns) +
      $(p.columns) * offs_h[:, None] + $(1) * offs_d[None, :]
    os_ptrs = OutScale + dest_index * $(p.heads) + $(1) * offs_h[:, None]
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = ((maximum / 127.0).to(OutScale.dtype.element_ty))[:, None]
    denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    tl.store(o_ptrs, q_src_data,
      mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.columns)))
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
  }


theorem original_body_eq (p : Parameters) :
    (kernel p).surfaceBody = (legacyKernel p).surfaceBody := by
  rfl

theorem candidate_body_eq (p : Parameters) :
    (addressEarly p).surfaceBody = (legacyEarly p).surfaceBody := by
  rfl

#axiomsClean original_body_eq
#axiomsClean candidate_body_eq
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelSource
