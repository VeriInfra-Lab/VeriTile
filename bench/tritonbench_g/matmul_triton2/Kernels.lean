import bench.tritonbench_g.matmul_triton2.Sources.Parameters
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001

def callSignature : CallSignature where
  pointers := [("a_ptr",.fp32),("b_ptr",.fp32),("c_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),
    ("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),
    ("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M"]

def body (a_ptr b_ptr c_ptr : RegionName)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    M = tl.cast(M, tl.uint64)
    N = tl.cast(N, tl.uint64)
    K = tl.cast(K, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_ak = tl.cast(stride_ak, tl.uint64)
    stride_bk = tl.cast(stride_bk, tl.uint64)
    stride_bn = tl.cast(stride_bn, tl.uint64)
    stride_cm = tl.cast(stride_cm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(axis=0).to(tl.uint64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.cast(min(num_pid_m - first_pid_m, GROUP_SIZE_M), tl.uint64)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.uint64)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.uint64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.uint64)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
  }
def scalarParameters (p : Core.Parameters) : List (RegName × Nat) :=
  [("M",p.rows),("N",p.columns),("K",p.innerSize),
    ("stride_am",p.inputRow),("stride_ak",p.inputInner),("stride_bk",p.weightInner),
    ("stride_bn",p.weightColumn),("stride_cm",p.outputRow),("stride_cn",p.outputColumn)]
noncomputable def arguments (p : Core.Parameters) : List ScalarArgument :=
  [ScalarArgument.integer "M" p.rows,ScalarArgument.integer "N" p.columns,
    ScalarArgument.integer "K" p.innerSize,ScalarArgument.integer "stride_am" p.inputRow,
    ScalarArgument.integer "stride_ak" p.inputInner,ScalarArgument.integer "stride_bk" p.weightInner,
    ScalarArgument.integer "stride_bn" p.weightColumn,ScalarArgument.integer "stride_cm" p.outputRow,
    ScalarArgument.integer "stride_cn" p.outputColumn]
noncomputable def source (p : Core.Parameters) : ComputeKernel :=
  (body p.input p.weight p.output p.blockRows p.blockColumns p.blockInner p.groupRows).withScalarArguments (arguments p)
/-- Reuse the loop-invariant row/column comparisons. -/
def maskHoistedBody (a_ptr b_ptr c_ptr : RegionName)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    M = tl.cast(M, tl.uint64)
    N = tl.cast(N, tl.uint64)
    K = tl.cast(K, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_ak = tl.cast(stride_ak, tl.uint64)
    stride_bk = tl.cast(stride_bk, tl.uint64)
    stride_bn = tl.cast(stride_bn, tl.uint64)
    stride_cm = tl.cast(stride_cm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(axis=0).to(tl.uint64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.cast(min(num_pid_m - first_pid_m, GROUP_SIZE_M), tl.uint64)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.uint64)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.uint64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.uint64)
    row_mask = offs_am < M
    column_mask = offs_bn < N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)) {
      a = tl.load(a_ptrs, mask=row_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
      b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & column_mask[None, :], other=0.0)
      accumulator += tl.dot(a, b)
      a_ptrs += BLOCK_SIZE_K * stride_ak
      b_ptrs += BLOCK_SIZE_K * stride_bk
    }
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = row_mask[:, None] & column_mask[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)
  }
noncomputable def maskHoisted (p : Core.Parameters) : ComputeKernel :=
  (maskHoistedBody p.input p.weight p.output p.blockRows p.blockColumns p.blockInner p.groupRows).withScalarArguments (arguments p)

end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
