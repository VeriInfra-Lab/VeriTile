/- Historical projected representation used by mathematical and byte-memory
bridges. Current public specifications execute Kernels.lean / CandidateKernels.lean
through the parameter bindings in the topic-level IO.lean. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural

/-! Readable quotations of the repaired JIT bodies. Geometry and integer
address casts use the shared exact-integer projection. Floating scratch
specializations retain their requested storage channel. The integer scratch
conversion is separately modeled at its scalar store boundary. Every source is
an ordinary complete quotation; no whole-source generation macro is used. -/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedSource
open VeriTile.Triton Lean

def stage1FP32 (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE program_base : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  triton {
        tl.ports inputs=[inp], outputs=[mid_value, Region.cast mid_index]
        tl.region inp = tl.float32, mid_value = tl.float32
        pid = tl.program_id(0) + $(program_base)
        if INT64_INDEX { pid = (pid).to(tl.int64) }
        offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        inp_ptrs = inp + offset
        mask = offset < $(M)
        inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = inp_val != inp_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), inp_val)
        max_val, max_index = tl.max(ordered_values, axis=0, return_indices=True)
        max_index = tl.where(has_nan, nan_index, max_index)
        max_index = max_index + pid * $(BLOCK_SIZE)
        max_val = tl.load(inp + max_index)
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val.to(tl.float32))
        tl.store(max_index_ptr, max_index.to(tl.int64))
      }

def stage1FP16 (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE program_base : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  triton {
        tl.ports inputs=[inp], outputs=[mid_value, Region.cast mid_index]
        tl.region inp = tl.float32, mid_value = tl.float16
        pid = tl.program_id(0) + $(program_base)
        if INT64_INDEX { pid = (pid).to(tl.int64) }
        offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        inp_ptrs = inp + offset
        mask = offset < $(M)
        inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = inp_val != inp_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), inp_val)
        max_val, max_index = tl.max(ordered_values, axis=0, return_indices=True)
        max_index = tl.where(has_nan, nan_index, max_index)
        max_index = max_index + pid * $(BLOCK_SIZE)
        max_val = tl.load(inp + max_index)
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val.to(tl.float16))
        tl.store(max_index_ptr, max_index.to(tl.int64))
      }

def stage1BF16 (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE program_base : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  triton {
        tl.ports inputs=[inp], outputs=[mid_value, Region.cast mid_index]
        tl.region inp = tl.float32, mid_value = tl.bfloat16
        pid = tl.program_id(0) + $(program_base)
        if INT64_INDEX { pid = (pid).to(tl.int64) }
        offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        inp_ptrs = inp + offset
        mask = offset < $(M)
        inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = inp_val != inp_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), inp_val)
        max_val, max_index = tl.max(ordered_values, axis=0, return_indices=True)
        max_index = tl.where(has_nan, nan_index, max_index)
        max_index = max_index + pid * $(BLOCK_SIZE)
        max_val = tl.load(inp + max_index)
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val.to(tl.bfloat16))
        tl.store(max_index_ptr, max_index.to(tl.int64))
      }

def stage1FP64 (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE program_base : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  triton {
        tl.ports inputs=[inp], outputs=[mid_value, Region.cast mid_index]
        tl.region inp = tl.float32, mid_value = tl.float64
        pid = tl.program_id(0) + $(program_base)
        if INT64_INDEX { pid = (pid).to(tl.int64) }
        offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        inp_ptrs = inp + offset
        mask = offset < $(M)
        inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = inp_val != inp_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), inp_val)
        max_val, max_index = tl.max(ordered_values, axis=0, return_indices=True)
        max_index = tl.where(has_nan, nan_index, max_index)
        max_index = max_index + pid * $(BLOCK_SIZE)
        max_val = tl.load(inp + max_index)
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val.to(tl.float64))
        tl.store(max_index_ptr, max_index.to(tl.int64))
      }

def stage1Integer (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE program_base : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  triton {
        tl.ports inputs=[inp], outputs=[mid_value, Region.cast mid_index]
        tl.region inp = tl.float32, mid_value = tl.int64
        pid = tl.program_id(0) + $(program_base)
        if INT64_INDEX { pid = (pid).to(tl.int64) }
        offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        inp_ptrs = inp + offset
        mask = offset < $(M)
        inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = inp_val != inp_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), inp_val)
        max_val, max_index = tl.max(ordered_values, axis=0, return_indices=True)
        max_index = tl.where(has_nan, nan_index, max_index)
        max_index = max_index + pid * $(BLOCK_SIZE)
        max_val = tl.load(inp + max_index)
        mid_value_ptr = mid_value + pid
        max_index_ptr = mid_index + pid
        tl.store(mid_value_ptr, max_val.to(tl.int64))
        tl.store(max_index_ptr, max_index.to(tl.int64))
      }

def stage2FP32 (mid_value : RegionName) (mid_index out : Region .int)
    (mid_size BLOCK_MID : Nat) : ComputeKernel :=
  triton {
        tl.region mid_value = tl.float32
        offset = tl.arange(0, $(BLOCK_MID))
        mid_ptrs = mid_value + offset
        mask = offset < $(mid_size)
        mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = mid_val != mid_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), mid_val)
        index_val = tl.argmax(ordered_values, axis=0)
        index_val = tl.where(has_nan, nan_index, index_val)
        mid_index_ptrs = mid_index + index_val
        out_val = tl.load(mid_index_ptrs)
        tl.store(out, out_val)
      }

def stage2FP16 (mid_value : RegionName) (mid_index out : Region .int)
    (mid_size BLOCK_MID : Nat) : ComputeKernel :=
  triton {
        tl.region mid_value = tl.float16
        offset = tl.arange(0, $(BLOCK_MID))
        mid_ptrs = mid_value + offset
        mask = offset < $(mid_size)
        mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = mid_val != mid_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), mid_val)
        index_val = tl.argmax(ordered_values, axis=0)
        index_val = tl.where(has_nan, nan_index, index_val)
        mid_index_ptrs = mid_index + index_val
        out_val = tl.load(mid_index_ptrs)
        tl.store(out, out_val)
      }

def stage2BF16 (mid_value : RegionName) (mid_index out : Region .int)
    (mid_size BLOCK_MID : Nat) : ComputeKernel :=
  triton {
        tl.region mid_value = tl.bfloat16
        offset = tl.arange(0, $(BLOCK_MID))
        mid_ptrs = mid_value + offset
        mask = offset < $(mid_size)
        mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = mid_val != mid_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), mid_val)
        index_val = tl.argmax(ordered_values, axis=0)
        index_val = tl.where(has_nan, nan_index, index_val)
        mid_index_ptrs = mid_index + index_val
        out_val = tl.load(mid_index_ptrs)
        tl.store(out, out_val)
      }

def stage2FP64 (mid_value : RegionName) (mid_index out : Region .int)
    (mid_size BLOCK_MID : Nat) : ComputeKernel :=
  triton {
        tl.region mid_value = tl.float64
        offset = tl.arange(0, $(BLOCK_MID))
        mid_ptrs = mid_value + offset
        mask = offset < $(mid_size)
        mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = mid_val != mid_val
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=0)
        has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
        ordered_values = tl.where(nan_lanes, -float("inf"), mid_val)
        index_val = tl.argmax(ordered_values, axis=0)
        index_val = tl.where(has_nan, nan_index, index_val)
        mid_index_ptrs = mid_index + index_val
        out_val = tl.load(mid_index_ptrs)
        tl.store(out, out_val)
      }
def stage2Integer (mid_value mid_index out : Region .int) (size block : Nat) (padding : Int) : ComputeKernel :=
  triton {
    offset = tl.arange(0,$(block))
    mid_ptrs = mid_value + offset
    mask = offset < $(size)
    mid_val = tl.load(mid_ptrs,mask=mask,other=tl.full([],dtype=tl.int64,value=$(padding)))
    index_val = tl.argmax(mid_val,axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out,out_val)
  }

def dimension (inp : RegionName) (out_index : Region .int)
    (M N K BLOCK_M BLOCK_N program_base : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  triton {
  tl.ports inputs=[inp], outputs=[Region.cast out_index]
    tl.region inp = tl.float32
    program = tl.program_id(0) + $(program_base)
    pid_m = program // $(K)
    pid_k = program % $(K)
    if INT64_INDEX {
      pid_m = (pid_m).to(tl.int64)
      pid_k = (pid_k).to(tl.int64)
    }
    m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    max_values = tl.full([$(BLOCK_M)], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([$(BLOCK_M)], dtype=tl.int64, value=0)
    for start_n in range($(0), $(N), $(BLOCK_N)) {
      n_offset = start_n + tl.arange(0, $(BLOCK_N))
      offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
      mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
      inp_ptrs = inp + offset
      inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
      nan_lanes = inp_vals != inp_vals
      nan_index = tl.argmax(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=1)
      nan_presence = tl.max(tl.where(nan_lanes, tl.full([], dtype=tl.int64, value=1), tl.full([], dtype=tl.int64, value=0)), axis=1)
      has_nan = nan_presence > tl.full([], dtype=tl.int64, value=0)
      ordered_values = tl.where(nan_lanes, -float("inf"), inp_vals)
      local_max, local_argmax = tl.max(ordered_values, 1,
        return_indices=True, return_indices_tie_break_left=True)
      local_argmax = tl.where(has_nan, nan_index, local_argmax)
      selected_column = start_n + local_argmax
      selected_offset = m_offset * $(N) * $(K) + selected_column * $(K) + pid_k
      selected_mask = m_offset < $(M) and selected_column < $(N)
      local_max = tl.load(inp + selected_offset, mask=selected_mask, other=-float("inf"))
      update = tl.logical_and(max_values == max_values, tl.logical_or(local_max != local_max, local_max > max_values))
      max_values = tl.where(update, local_max, max_values)
      argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
    }
    offset_index = m_offset * $(K) + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < $(M)
    tl.store(out_index_ptrs, argmax_values, mask=mask1)
  }
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedSource
