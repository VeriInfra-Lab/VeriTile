/-
Source-audit target: max_kernel_1
Source-audit Lean: stage1
Source-audit args: dtype=tl.float32, version=original
Selected lexical source view only; this metadata is not all-profile certification.
Original fp32 first-stage maximum quotation, including its masked infinity padding.
max_kernel_2, max_kernel, other dtypes and candidate transformations are unselected.
-/

import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def max_kernel_1
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_2`.

Same allowed Lean-syntax-only changes as `max_kernel_1`. -/
def max_kernel_2
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel`
(autotuned, returns value + index via `tl.max(..., return_indices=True)`).

Allowed mechanical Lean-syntax-only changes apply. The `@triton.autotune`
and `@triton.heuristics` decorators that wrap `max_kernel` are not DSL-side
constructs (they configure the launch grid, not the kernel body); they have
no in-kernel transcription. -/
def max_kernel
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
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
  tl.store(out_index_ptrs, result_index, mask=mask1)
}


/-! Source-facing value reductions load their declared storage, retain fp64 or
promote narrower formats to fp32, and narrow once when writing half/bfloat/fp8 storage.
The first candidate commutes an index product; the second complements the
integer bound comparison, preserving the exact active set. -/
inductive Storage where
  | fp32 | fp16 | bf16 | f8e4 | f8e5 | fp64
  deriving DecidableEq

abbrev ValueSource := RegionName → RegionName → Nat → Nat → ComputeKernel

/-- Complete ordinary source for each retained storage and existing index candidate.
No runtime dtype or branch is removed by this source-only expansion. -/
def stage1 (storage : Storage) (candidate : Bool) : ValueSource :=
  match storage,candidate with
  | .fp32,Bool.false => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
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
  | .fp32,Bool.true => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
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
  | .fp16,Bool.false => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float16, mid = tl.float16
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.float16))
}
  | .fp16,Bool.true => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float16, mid = tl.float16
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.float16))
}
  | .bf16,Bool.false => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.bfloat16, mid = tl.bfloat16
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.bfloat16))
}
  | .bf16,Bool.true => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.bfloat16, mid = tl.bfloat16
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.bfloat16))
}
  | .f8e4,Bool.false => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float8e4nv, mid = tl.float8e4nv
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.float8e4nv))
}
  | .f8e4,Bool.true => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float8e4nv, mid = tl.float8e4nv
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.float8e4nv))
}
  | .f8e5,Bool.false => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float8e5, mid = tl.float8e5
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.float8e5))
}
  | .f8e5,Bool.true => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float8e5, mid = tl.float8e5
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val.to(tl.float8e5))
}
  | .fp64,Bool.false => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float64, mid = tl.float64
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}
  | .fp64,Bool.true => fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => triton {
  tl.region inp = tl.float64, mid = tl.float64
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}

def stage2 (storage : Storage) (candidate : Bool) : ValueSource :=
  match storage,candidate with
  | .fp32,Bool.false => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float32, out = tl.float32
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}
  | .fp32,Bool.true => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float32, out = tl.float32
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}
  | .fp16,Bool.false => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float16, out = tl.float16
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.float16))
}
  | .fp16,Bool.true => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float16, out = tl.float16
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.float16))
}
  | .bf16,Bool.false => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.bfloat16, out = tl.bfloat16
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.bfloat16))
}
  | .bf16,Bool.true => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.bfloat16, out = tl.bfloat16
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.bfloat16))
}
  | .f8e4,Bool.false => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float8e4nv, out = tl.float8e4nv
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.float8e4nv))
}
  | .f8e4,Bool.true => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float8e4nv, out = tl.float8e4nv
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.float8e4nv))
}
  | .f8e5,Bool.false => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float8e5, out = tl.float8e5
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.float8e5))
}
  | .f8e5,Bool.true => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float8e5, out = tl.float8e5
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val.to(tl.float8e5))
}
  | .fp64,Bool.false => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float64, out = tl.float64
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}
  | .fp64,Bool.true => fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => triton {
  tl.region mid = tl.float64, out = tl.float64
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}

/-- Complete dimension-reduction source, including both masked typed output
streams and the explicit signed-int64 result-index store. -/
def dimensionSource (storage : Storage) (candidate : Bool)
    (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  match storage,candidate with
  | .fp32,Bool.false => triton {
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
  | .fp32,Bool.true => triton {
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
  | .fp16,Bool.false => triton {
  tl.region inp = tl.float16, out_value = tl.float16, out_index = tl.int64
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
  | .fp16,Bool.true => triton {
  tl.region inp = tl.float16, out_value = tl.float16, out_index = tl.int64
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
  | .bf16,Bool.false => triton {
  tl.region inp = tl.bfloat16, out_value = tl.bfloat16, out_index = tl.int64
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
  tl.store(out_value_ptrs, result_value.to(tl.bfloat16), mask=mask1)
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}
  | .bf16,Bool.true => triton {
  tl.region inp = tl.bfloat16, out_value = tl.bfloat16, out_index = tl.int64
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
  tl.store(out_value_ptrs, result_value.to(tl.bfloat16), mask=mask1)
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}
  | .f8e4,Bool.false => triton {
  tl.region inp = tl.float8e4nv, out_value = tl.float8e4nv, out_index = tl.int64
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
  | .f8e4,Bool.true => triton {
  tl.region inp = tl.float8e4nv, out_value = tl.float8e4nv, out_index = tl.int64
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
  | .f8e5,Bool.false => triton {
  tl.region inp = tl.float8e5, out_value = tl.float8e5, out_index = tl.int64
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
  | .f8e5,Bool.true => triton {
  tl.region inp = tl.float8e5, out_value = tl.float8e5, out_index = tl.int64
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
  | .fp64,Bool.false => triton {
  tl.region inp = tl.float64, out_value = tl.float64, out_index = tl.int64
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
  | .fp64,Bool.true => triton {
  tl.region inp = tl.float64, out_value = tl.float64, out_index = tl.int64
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

/- Compatibility syntax delegates to complete ordinary sources. It generates
no statements, storage casts, guards or source AST fragments. -/
syntax "maxStage1%" "[" tritonDType "," ident "]" : term
syntax "maxStage2%" "[" tritonDType "," ident "]" : term
syntax "maxDimension%" "[" tritonDType "," ident "]" : term
macro_rules
  | `(maxStage1%[tl.float32,original]) => `(stage1 .fp32 Bool.false)
  | `(maxStage2%[tl.float32,original]) => `(stage2 .fp32 Bool.false)
  | `(maxDimension%[tl.float32,original]) => `(dimensionSource .fp32 Bool.false)
  | `(maxStage1%[tl.float32,indexCommuted]) => `(stage1 .fp32 Bool.true)
  | `(maxStage2%[tl.float32,maskComplement]) => `(stage2 .fp32 Bool.true)
  | `(maxDimension%[tl.float32,indexCommuted]) => `(dimensionSource .fp32 Bool.true)
  | `(maxStage1%[tl.float16,original]) => `(stage1 .fp16 Bool.false)
  | `(maxStage2%[tl.float16,original]) => `(stage2 .fp16 Bool.false)
  | `(maxDimension%[tl.float16,original]) => `(dimensionSource .fp16 Bool.false)
  | `(maxStage1%[tl.float16,indexCommuted]) => `(stage1 .fp16 Bool.true)
  | `(maxStage2%[tl.float16,maskComplement]) => `(stage2 .fp16 Bool.true)
  | `(maxDimension%[tl.float16,indexCommuted]) => `(dimensionSource .fp16 Bool.true)
  | `(maxStage1%[tl.bfloat16,original]) => `(stage1 .bf16 Bool.false)
  | `(maxStage2%[tl.bfloat16,original]) => `(stage2 .bf16 Bool.false)
  | `(maxDimension%[tl.bfloat16,original]) => `(dimensionSource .bf16 Bool.false)
  | `(maxStage1%[tl.bfloat16,indexCommuted]) => `(stage1 .bf16 Bool.true)
  | `(maxStage2%[tl.bfloat16,maskComplement]) => `(stage2 .bf16 Bool.true)
  | `(maxDimension%[tl.bfloat16,indexCommuted]) => `(dimensionSource .bf16 Bool.true)
  | `(maxStage1%[tl.float8e4nv,original]) => `(stage1 .f8e4 Bool.false)
  | `(maxStage2%[tl.float8e4nv,original]) => `(stage2 .f8e4 Bool.false)
  | `(maxDimension%[tl.float8e4nv,original]) => `(dimensionSource .f8e4 Bool.false)
  | `(maxStage1%[tl.float8e4nv,indexCommuted]) => `(stage1 .f8e4 Bool.true)
  | `(maxStage2%[tl.float8e4nv,maskComplement]) => `(stage2 .f8e4 Bool.true)
  | `(maxDimension%[tl.float8e4nv,indexCommuted]) => `(dimensionSource .f8e4 Bool.true)
  | `(maxStage1%[tl.float8e5,original]) => `(stage1 .f8e5 Bool.false)
  | `(maxStage2%[tl.float8e5,original]) => `(stage2 .f8e5 Bool.false)
  | `(maxDimension%[tl.float8e5,original]) => `(dimensionSource .f8e5 Bool.false)
  | `(maxStage1%[tl.float8e5,indexCommuted]) => `(stage1 .f8e5 Bool.true)
  | `(maxStage2%[tl.float8e5,maskComplement]) => `(stage2 .f8e5 Bool.true)
  | `(maxDimension%[tl.float8e5,indexCommuted]) => `(dimensionSource .f8e5 Bool.true)
  | `(maxStage1%[tl.float64,original]) => `(stage1 .fp64 Bool.false)
  | `(maxStage2%[tl.float64,original]) => `(stage2 .fp64 Bool.false)
  | `(maxDimension%[tl.float64,original]) => `(dimensionSource .fp64 Bool.false)
  | `(maxStage1%[tl.float64,indexCommuted]) => `(stage1 .fp64 Bool.true)
  | `(maxStage2%[tl.float64,maskComplement]) => `(stage2 .fp64 Bool.true)
  | `(maxDimension%[tl.float64,indexCommuted]) => `(dimensionSource .fp64 Bool.true)
end VeriTile.Bench.TritonBenchG.MaxReduction
