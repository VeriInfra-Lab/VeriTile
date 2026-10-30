import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.matmul_persistent_triton.Types
/-! Literal original/candidate Python bodies for both wrapper storage profiles.
Pointer dtypes and runtime scalar bindings are call metadata outside the body.
The row-index commutation is the only candidate transformation. -/
namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TypedSource
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource.Storage
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

def fp16Signature : CallSignature where
  pointers := [("a_ptr", .fp16), ("b_ptr", .fp16), ("c_ptr", .fp16)]
  scalars := [("M",.i64), ("N",.i64), ("K",.i64), ("stride_am",.i64), ("stride_ak",.i64), ("stride_bk",.i64), ("stride_bn",.i64), ("stride_cm",.i64), ("stride_cn",.i64)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_SMS"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def fp16 (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) : ComputeKernel :=
  with_triton_call fp16Signature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
    start_pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0)
    k_tiles = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS + (start_pid < num_tiles % NUM_SMS)
    active_group_m = min(GROUP_SIZE_M, num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for tile_ordinal in range(0, tiles_per_SM):
        tile_id = start_pid + tile_ordinal * NUM_SMS
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * active_group_m
        group_size_m = min(num_pid_m - first_pid_m, active_group_m)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_cm = start_m + lanes_m
        offs_cn = start_n + lanes_n
        offs_am = tl.where(offs_cm < M, offs_cm, 0)
        offs_bn = tl.where(offs_cn < N, offs_cn, 0)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(0, k_tiles):
            raw_k = ki * BLOCK_SIZE_K + offs_k_for_mask
            offs_k = tl.where(raw_k < K, raw_k, 0)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)
  }

def f8e4Signature : CallSignature where
  pointers := [("a_ptr", .f8e4), ("b_ptr", .f8e4), ("c_ptr", .f8e4)]
  scalars := [("M",.i64), ("N",.i64), ("K",.i64), ("stride_am",.i64), ("stride_ak",.i64), ("stride_bk",.i64), ("stride_bn",.i64), ("stride_cm",.i64), ("stride_cn",.i64)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_SMS"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def f8e4 (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) : ComputeKernel :=
  with_triton_call f8e4Signature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
    start_pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0)
    k_tiles = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS + (start_pid < num_tiles % NUM_SMS)
    active_group_m = min(GROUP_SIZE_M, num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for tile_ordinal in range(0, tiles_per_SM):
        tile_id = start_pid + tile_ordinal * NUM_SMS
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * active_group_m
        group_size_m = min(num_pid_m - first_pid_m, active_group_m)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_cm = start_m + lanes_m
        offs_cn = start_n + lanes_n
        offs_am = tl.where(offs_cm < M, offs_cm, 0)
        offs_bn = tl.where(offs_cn < N, offs_cn, 0)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(0, k_tiles):
            raw_k = ki * BLOCK_SIZE_K + offs_k_for_mask
            offs_k = tl.where(raw_k < K, raw_k, 0)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)
  }

def fp16ReorderedSignature : CallSignature where
  pointers := [("a_ptr", .fp16), ("b_ptr", .fp16), ("c_ptr", .fp16)]
  scalars := [("M",.i64), ("N",.i64), ("K",.i64), ("stride_am",.i64), ("stride_ak",.i64), ("stride_bk",.i64), ("stride_bn",.i64), ("stride_cm",.i64), ("stride_cn",.i64)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_SMS"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def fp16Reordered (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) : ComputeKernel :=
  with_triton_call fp16ReorderedSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
    start_pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0)
    k_tiles = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS + (start_pid < num_tiles % NUM_SMS)
    active_group_m = min(GROUP_SIZE_M, num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for tile_ordinal in range(0, tiles_per_SM):
        tile_id = start_pid + tile_ordinal * NUM_SMS
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * active_group_m
        group_size_m = min(num_pid_m - first_pid_m, active_group_m)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        start_m = BLOCK_SIZE_M * pid_m
        start_n = pid_n * BLOCK_SIZE_N
        offs_cm = start_m + lanes_m
        offs_cn = start_n + lanes_n
        offs_am = tl.where(offs_cm < M, offs_cm, 0)
        offs_bn = tl.where(offs_cn < N, offs_cn, 0)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(0, k_tiles):
            raw_k = ki * BLOCK_SIZE_K + offs_k_for_mask
            offs_k = tl.where(raw_k < K, raw_k, 0)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)
  }

def f8e4ReorderedSignature : CallSignature where
  pointers := [("a_ptr", .f8e4), ("b_ptr", .f8e4), ("c_ptr", .f8e4)]
  scalars := [("M",.i64), ("N",.i64), ("K",.i64), ("stride_am",.i64), ("stride_ak",.i64), ("stride_bk",.i64), ("stride_bn",.i64), ("stride_cm",.i64), ("stride_cn",.i64)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_SMS"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def f8e4Reordered (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NUM_SMS : Nat) : ComputeKernel :=
  with_triton_call f8e4ReorderedSignature call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
    start_pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0)
    k_tiles = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS + (start_pid < num_tiles % NUM_SMS)
    active_group_m = min(GROUP_SIZE_M, num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for tile_ordinal in range(0, tiles_per_SM):
        tile_id = start_pid + tile_ordinal * NUM_SMS
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * active_group_m
        group_size_m = min(num_pid_m - first_pid_m, active_group_m)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        start_m = BLOCK_SIZE_M * pid_m
        start_n = pid_n * BLOCK_SIZE_N
        offs_cm = start_m + lanes_m
        offs_cn = start_n + lanes_n
        offs_am = tl.where(offs_cm < M, offs_cm, 0)
        offs_bn = tl.where(offs_cn < N, offs_cn, 0)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(0, k_tiles):
            raw_k = ki * BLOCK_SIZE_K + offs_k_for_mask
            offs_k = tl.where(raw_k < K, raw_k, 0)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)
  }
def kernel : Storage → VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource.Source
  | .fp16 => fp16
  | .f8e4 => f8e4

/-- Reuse the predecessor row-index multiplication canonicalization at its
selected common row-coordinate calculation. Every use gets the same integer;
all floating operations, masks and persistent scheduling remain unchanged. -/
def reordered : Storage → VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.TypedSource.Source
  | .fp16 => fp16Reordered
  | .f8e4 => f8e4Reordered

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TypedSource
