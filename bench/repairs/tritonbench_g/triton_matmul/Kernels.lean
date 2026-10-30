import bench.tritonbench_g.triton_matmul.Kernels
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
noncomputable section

namespace Source
def signature_fp16 : CallSignature where
  pointers := [("a_ptr",.fp16), ("b_ptr",.fp16), ("c_ptr",.fp16)]
  scalars := [("M",.i64), ("N",.i64), ("K",.i64), ("stride_am",.i64), ("stride_ak",.i64), ("stride_bk",.i64), ("stride_bn",.i64), ("stride_cm",.i64), ("stride_cn",.i64)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"]
  readPorts := ["a_ptr", "b_ptr"]
  writePorts := ["c_ptr"]

def halfSource (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_call signature_fp16 call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
    # Widen before all dimension, offset, and stride arithmetic.
    pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    # Quotient/remainder ceiling avoids overflowing dimension + block - 1.
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0).to(tl.int64)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    mask_m = lanes_m < M - start_m
    mask_n = lanes_n < N - start_n
    # Clamp both addends before adding, including inactive output lanes.
    offs_am = tl.where(mask_m, start_m, 0) + tl.where(mask_m, lanes_m, 0)
    offs_bn = tl.where(mask_n, start_n, 0) + tl.where(mask_n, lanes_n, 0)

    # Clamped tail coordinates do not promise full-tile contiguity.
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    blocks_k = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0).to(tl.int64)
    for k in range(0, blocks_k):
        base_k = k * BLOCK_SIZE_K
        mask_k = offs_k < K - base_k
        safe_k = tl.where(mask_k, base_k, 0) + tl.where(mask_k, offs_k, 0)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=c_mask)
  }

def signature_f8e4 : CallSignature where
  pointers := [("a_ptr",.f8e4), ("b_ptr",.f8e4), ("c_ptr",.f8e4)]
  scalars := [("M",.i64), ("N",.i64), ("K",.i64), ("stride_am",.i64), ("stride_ak",.i64), ("stride_bk",.i64), ("stride_bn",.i64), ("stride_cm",.i64), ("stride_cn",.i64)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"]
  readPorts := ["a_ptr", "b_ptr"]
  writePorts := ["c_ptr"]

def byteSource (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_call signature_f8e4 call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in
  triton {
    # Widen before all dimension, offset, and stride arithmetic.
    pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    # Quotient/remainder ceiling avoids overflowing dimension + block - 1.
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0).to(tl.int64)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0).to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    mask_m = lanes_m < M - start_m
    mask_n = lanes_n < N - start_n
    # Clamp both addends before adding, including inactive output lanes.
    offs_am = tl.where(mask_m, start_m, 0) + tl.where(mask_m, lanes_m, 0)
    offs_bn = tl.where(mask_n, start_n, 0) + tl.where(mask_n, lanes_n, 0)

    # Clamped tail coordinates do not promise full-tile contiguity.
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    blocks_k = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0).to(tl.int64)
    for k in range(0, blocks_k):
        base_k = k * BLOCK_SIZE_K
        mask_k = offs_k < K - base_k
        safe_k = tl.where(mask_k, base_k, 0) + tl.where(mask_k, offs_k, 0)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=c_mask)
  }

/-- The host selects storage and supplies its original argument values. The
quoted functions above contain only the unmodified Python function body. -/
def typed (p : Parameters) (st : Storage) : ComputeKernel :=
  match st with
  | .half => halfSource p.A p.B p.C p.M p.N p.K
      p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn
      (BM p) (BN p) (BK p) p.GROUP_SIZE_M
  | .e4m3 => byteSource p.A p.B p.C p.M p.N p.K
      p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn
      (BM p) (BN p) (BK p) p.GROUP_SIZE_M

end Source
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
