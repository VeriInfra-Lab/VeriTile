import bench.repairs.tritonbench_g.quantize_kv_copy.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelSyntax
open VeriTile.Triton
set_option maxHeartbeats 3000000
private def legacyKernel (groups block heads : Nat) : ComputeKernel :=
  let K : RegionName := "K"
  let Dest_loc : Region .int := ⟨"D"⟩
  let Out : RegionName := "O"
  let Out_scale : RegionName := "S"
  triton {
    tl.region K = tl.float32, Out = tl.int8, Out_scale = tl.float32
    cur_index = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_g = tl.arange(0, $(block)).to(tl.int64)
    offs_d = tl.arange(0, $(8)).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * $(1)).to(tl.int64)
    src_data = tl.load(K + cur_index * $(heads*groups*8) + cur_head * $(groups*8)
      + offs_g[:, None] * $(8) + offs_d[None, :] * $(1),
      mask=offs_g[:, None] < $(groups), other=0.0)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(Out_scale.dtype.element_ty)
    denominator = tl.where(maximum == 0.0, 1.0, data_scale)
    q_src_data = (src_data / denominator[:, None]).to(tl.int8)
    o_ptrs = Out + dest_index * $(heads*groups*8) + cur_head * $(groups*8)
      + offs_g[:, None] * $(8) + offs_d[None, :] * $(1)
    os_ptrs = Out_scale + dest_index * $(heads*groups) + cur_head * $(groups) + offs_g * $(1)
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < $(groups))
    tl.store(os_ptrs, data_scale, mask=offs_g < $(groups))
  }

/-- Schedule pure output address formation before floating quantization to expose
instruction overlap. Floating instructions, masks and stores are unchanged. -/
private def legacyEarly (groups block heads : Nat) : ComputeKernel :=
  let K : RegionName := "K"
  let Dest_loc : Region .int := ⟨"D"⟩
  let Out : RegionName := "O"
  let Out_scale : RegionName := "S"
  triton {
    tl.region K = tl.float32, Out = tl.int8, Out_scale = tl.float32
    cur_index = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_g = tl.arange(0, $(block)).to(tl.int64)
    offs_d = tl.arange(0, $(8)).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * $(1)).to(tl.int64)
    src_data = tl.load(K + cur_index * $(heads*groups*8) + cur_head * $(groups*8)
      + offs_g[:, None] * $(8) + offs_d[None, :] * $(1),
      mask=offs_g[:, None] < $(groups), other=0.0)
    o_ptrs = Out + dest_index * $(heads*groups*8) + cur_head * $(groups*8)
      + offs_g[:, None] * $(8) + offs_d[None, :] * $(1)
    os_ptrs = Out_scale + dest_index * $(heads*groups) + cur_head * $(groups) + offs_g * $(1)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(Out_scale.dtype.element_ty)
    denominator = tl.where(maximum == 0.0, 1.0, data_scale)
    q_src_data = (src_data / denominator[:, None]).to(tl.int8)
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < $(groups))
    tl.store(os_ptrs, data_scale, mask=offs_g < $(groups))
  }


theorem original_body_eq (groups block heads : Nat) :
    (source groups block heads).surfaceBody = (legacyKernel groups block heads).surfaceBody := by
  rfl

theorem candidate_body_eq (groups block heads : Nat) :
    (addressEarly groups block heads).surfaceBody = (legacyEarly groups block heads).surfaceBody := by
  rfl

#axiomsClean original_body_eq
#axiomsClean candidate_body_eq
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelSyntax
