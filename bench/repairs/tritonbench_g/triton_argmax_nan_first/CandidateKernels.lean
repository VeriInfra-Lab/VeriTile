import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Kernels

/-! Complete original Python bodies. Runtime scalars, source pointer dtypes
and constexpr arguments belong to the call signature outside the quotation. -/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITCandidate
open VeriTile.Triton

def stage1Signature : CallSignature where
  pointers := [("inp",.fp32),("mid_index",.i64)]
  sourceDTypePointers := ["mid_value"]
  scalars := [("M",.i64),("program_base",.i64)]
  constexpr := ["BLOCK_SIZE","INT64_INDEX"]
  readPorts := ["inp"]
  writePorts := ["mid_value","mid_index"]

def stage1 (inp mid_value : RegionName) (mid_index : Region .int)
    (MID_DTYPE : SourceDType) (BLOCK_SIZE : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  with_triton_signature stage1Signature source_dtype mid_value := MID_DTYPE in triton {
    pid = tl.program_id(0).to(tl.int64) + program_base
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    if BLOCK_SIZE == 1024:
        offset = (pid.to(tl.int64) << 10) + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    else:
        offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    nan_lanes = inp_val != inp_val
    nan_index = tl.argmax(tl.where(nan_lanes, tl.full((), 1, tl.int64), tl.full((), 0, tl.int64)), axis=0)
    has_nan = tl.max(tl.where(nan_lanes, tl.full((), 1, tl.int64), tl.full((), 0, tl.int64)), axis=0) > tl.full((), 0, tl.int64)
    ordered_values = tl.where(nan_lanes, -float("inf"), inp_val)
    max_val, max_index = tl.max(ordered_values, axis=0, return_indices=True)
    max_index = tl.where(has_nan, nan_index, max_index)
    max_index = max_index.to(tl.int64) + pid.to(tl.int64) * BLOCK_SIZE
    max_val = tl.load(inp + max_index)
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    stored_value = max_val
    if mid_value.dtype.element_ty.is_int():
        is_bool: tl.constexpr = mid_value.dtype.element_ty.is_int1()
        width: tl.constexpr = 8 if is_bool else mid_value.dtype.element_ty.primitive_bitwidth
        signed: tl.constexpr = is_bool or mid_value.dtype.element_ty.is_int_signed()
        exponent: tl.constexpr = width - 1 if signed else width
        lower: tl.constexpr = -(1 << exponent) if signed else 0
        gap: tl.constexpr = (1 << (exponent - 24)) if exponent > 24 else 1
        upper: tl.constexpr = (1 << exponent) - gap
        stored_value = tl.where(max_val == max_val, max_val, 0.0)
        stored_value = tl.minimum(
            tl.maximum(stored_value, tl.full((), lower, tl.float32)),
            tl.full((), upper, tl.float32),
        )
    tl.store(mid_value_ptr, stored_value)
    tl.store(max_index_ptr, max_index)
  }

def dimensionSignature : CallSignature where
  pointers := [("inp",.fp32),("out_index",.i64)]
  scalars := [("M",.i64),("N",.i64),("K",.i64),("program_base",.i64)]
  constexpr := ["BLOCK_M","BLOCK_N","INT64_INDEX"]
  readPorts := ["inp"]
  writePorts := ["out_index"]

def dimension (inp : RegionName) (out_index : Region .int)
    (BLOCK_M BLOCK_N : Nat) (INT64_INDEX : Bool) : ComputeKernel :=
  with_triton_signature dimensionSignature in triton {
    program = tl.program_id(0).to(tl.int64) + program_base
    pid_m = program // K
    pid_k = program % K
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)
    if BLOCK_M == 128:
        m_offset = (pid_m.to(tl.int64) << 7) + tl.arange(0, BLOCK_M).to(tl.int64)
    else:
        m_offset = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)

    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k.to(tl.int64)
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        nan_lanes = inp_vals != inp_vals
        nan_index = tl.argmax(tl.where(nan_lanes, tl.full((), 1, tl.int64), tl.full((), 0, tl.int64)), axis=1)
        has_nan = tl.max(tl.where(nan_lanes, tl.full((), 1, tl.int64), tl.full((), 0, tl.int64)), axis=1) > tl.full((), 0, tl.int64)
        ordered_values = tl.where(nan_lanes, -float("inf"), inp_vals)
        local_max, local_argmax = tl.max(
            ordered_values, 1, return_indices=True, return_indices_tie_break_left=True
        )
        local_argmax = tl.where(has_nan, nan_index, local_argmax)
        selected_column = start_n + local_argmax.to(tl.int64)
        selected_offset = m_offset * N * K + selected_column * K + pid_k.to(tl.int64)
        selected_mask = (m_offset < M) & (selected_column < N)
        local_max = tl.load(inp + selected_offset, mask=selected_mask, other=-float("inf"))
        update = (max_values == max_values) & ((local_max != local_max) | (local_max > max_values))
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax.to(tl.int64), argmax_values)

    offset_index = m_offset * K + pid_k.to(tl.int64)
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)
  }

end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITCandidate
