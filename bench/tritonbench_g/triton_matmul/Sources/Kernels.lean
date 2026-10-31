import bench.tritonbench_g.triton_matmul.Sources.Parameters
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
set_option maxRecDepth 32000
set_option maxHeartbeats 2000000

def signature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("a_ptr",dtype),("b_ptr",dtype),("c_ptr",dtype)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),
    ("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),("stride_bn",.integer),
    ("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M"]

def rawBody (dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_signature (signature dtype) in triton {
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

def call (dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  (rawBody dtype a_ptr b_ptr c_ptr BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M).withScalarArguments
    [ScalarArgument.integer "M" M,ScalarArgument.integer "N" N,ScalarArgument.integer "K" K,
     ScalarArgument.integer "stride_am" stride_am,ScalarArgument.integer "stride_ak" stride_ak,
     ScalarArgument.integer "stride_bk" stride_bk,ScalarArgument.integer "stride_bn" stride_bn,
     ScalarArgument.integer "stride_cm" stride_cm,ScalarArgument.integer "stride_cn" stride_cn]

def typed (p : Parameters) (st : Storage) : ComputeKernel :=
  call (dtype st) p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn
    p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
