import bench.tritonbench_g.max_reduction.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Unchanged complete selected Python bodies and the existing three integer
variants. Source casts remain in the quotations; dtype/ABI/call metadata is
outside. Mathematical targets and machine bounds are independently retained. -/
namespace VeriTile.Bench.Repairs.MaxReduction
open VeriTile.Triton

def stage1Signature : CallSignature where
  pointers := [("inp",.fp32), ("mid",.fp32)]
  scalars := [("M",.i32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["inp"]
  writePorts := ["mid"]

def stage1 (inp mid : RegionName) (M BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call stage1Signature call_args [M := M] in
  triton {
    pid = tl.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)
  }

def stage1Commuted (inp mid : RegionName) (M BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call stage1Signature call_args [M := M] in
  triton {
    pid = tl.program_id(0).to(tl.int64)
    offset = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)
  }

def stage2Signature : CallSignature where
  pointers := [("mid",.fp32), ("out",.fp32)]
  scalars := [("mid_size",.i32)]
  constexpr := ["BLOCK_MID"]
  readPorts := ["mid"]
  writePorts := ["out"]

def stage2 (mid out : RegionName) (mid_size BLOCK_MID : Nat) : ComputeKernel :=
  with_triton_call stage2Signature call_args [mid_size := mid_size] in
  triton {
    offset = tl.arange(0, BLOCK_MID).to(tl.int64)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)
  }

def stage2Complement (mid out : RegionName) (mid_size BLOCK_MID : Nat) : ComputeKernel :=
  with_triton_call stage2Signature call_args [mid_size := mid_size] in
  triton {
    offset = tl.arange(0, BLOCK_MID).to(tl.int64)
    mid_ptrs = mid + offset
    mask = not (mid_size <= offset)
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)
  }

def dimensionSignature : CallSignature where
  pointers := [("inp",.fp32), ("out_value",.fp32), ("out_index",.i64)]
  scalars := [("M",.i32), ("N",.i32), ("K",.i32)]
  constexpr := ["BLOCK_M", "BLOCK_N"]
  readPorts := ["inp"]
  writePorts := ["out_value", "out_index"]

def dimension (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call dimensionSignature call_args [M := M, N := N, K := K] in
  triton {
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    offset = m_offset[:, None] * tl.cast(N, tl.int64) * tl.cast(K, tl.int64) + n_offset[None, :] * tl.cast(K, tl.int64) + pid_k
    offset_index = m_offset * tl.cast(K, tl.int64) + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)
  }

def dimensionCommuted (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call dimensionSignature call_args [M := M, N := N, K := K] in
  triton {
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = BLOCK_M * pid_m + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    offset = m_offset[:, None] * tl.cast(N, tl.int64) * tl.cast(K, tl.int64) + n_offset[None, :] * tl.cast(K, tl.int64) + pid_k
    offset_index = m_offset * tl.cast(K, tl.int64) + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)
  }

end VeriTile.Bench.Repairs.MaxReduction
