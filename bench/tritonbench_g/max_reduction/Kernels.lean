import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure

/-! The repaired FP32 max programs and their existing integer-index candidates.
The scalar reduction has two real source stages; dimension reduction writes
both maximum values and signed indices. All stages use these shared bodies. -/
namespace VeriTile.Bench.Repairs.MaxReduction
open VeriTile.Triton

def stage1 (inp mid : RegionName) (M BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region inp = tl.float32, mid = tl.float32
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}

def stage1Commuted (inp mid : RegionName) (M BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region inp = tl.float32, mid = tl.float32
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}

def stage2 (mid out : RegionName) (mid_size BLOCK_MID : Nat) : ComputeKernel := triton {
  tl.region mid = tl.float32, out = tl.float32
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}

def stage2Complement (mid out : RegionName) (mid_size BLOCK_MID : Nat) : ComputeKernel := triton {
  tl.region mid = tl.float32, out = tl.float32
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}

def dimension (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel := triton {
  tl.region inp = tl.float32, out_value = tl.float32, out_index = tl.int64
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}

def dimensionCommuted (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel := triton {
  tl.region inp = tl.float32, out_value = tl.float32, out_index = tl.int64
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}

end VeriTile.Bench.Repairs.MaxReduction
