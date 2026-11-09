import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.Source

/-! Literal repaired INT8 row-wise dequantizing matmul and its existing index
reuse candidate. Both source programs retain all integer casts, original
variable reassignments, dynamic reduction bounds, bias and split-K branches.
Pointer/scalar storage and the unused ACC_TYPE constexpr belong to the caller. -/
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.TypedSource
open VeriTile.Triton
noncomputable section

def callSignature : CallSignature where
  pointers := [("A",.i8),("B",.i8),("C",.fp16),("bias",.fp32),
    ("state_x_ptr",.fp32),("state_w_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),("divfactor",.fp32),
    ("stride_am",.integer),("stride_ak",.integer),("stride_bk",.integer),
    ("stride_bn",.integer),("stride_cm",.integer),("stride_cn",.integer)]
  constexpr := ["has_bias","BLOCK_M","BLOCK_N","BLOCK_K","GROUP_M","SPLIT_K","EVEN_K","ACC_TYPE"]
  readPorts := ["A","B","bias","state_x_ptr","state_w_ptr"]
  writePorts := ["C"]

def sourceKernel (A B C bias state_x_ptr state_w_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K : Nat)
    (EVEN_K has_bias : Bool) : ComputeKernel :=
  let ACC_TYPE : String := "tl.int32"
  with_triton_call callSignature call_args [M := M, N := N, K := K,
    stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk,
    stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    pid = tl.program_id(0).to(tl.int64)
    pid_z = tl.program_id(1).to(tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w_factor = tl.load(state_w_ptr + rbn)[None, :]
    x_factor = tl.load(state_x_ptr + ram)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = w_factor * (x_factor * (acc * divfactor))
    acc = acc.to(C.dtype.element_ty)
    if has_bias:
        if pid_z == 0:
            bias_value = tl.load(bias + rn, mask=rn < N, other=0.0).to(C.dtype.element_ty)
            acc = acc + bias_value[None, :]
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
  }

def candidateKernel (A B C bias state_x_ptr state_w_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K : Nat)
    (EVEN_K has_bias : Bool) : ComputeKernel :=
  let ACC_TYPE : String := "tl.int32"
  with_triton_call callSignature call_args [M := M, N := N, K := K,
    stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk,
    stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn] in triton {
    pid = tl.program_id(0).to(tl.int64)
    pid_z = tl.program_id(1).to(tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    w_factor = tl.load(state_w_ptr + rbn)[None, :]
    x_factor = tl.load(state_x_ptr + ram)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = w_factor * (x_factor * (acc * divfactor))
    acc = acc.to(C.dtype.element_ty)
    if has_bias:
        if pid_z == 0:
            bias_value = tl.load(bias + rn, mask=rn < N, other=0.0).to(C.dtype.element_ty)
            acc = acc + bias_value[None, :]
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
  }

/-- Retained API: the source itself computes its reduction bound from K and
its launch constexprs. The old redundant steps argument does not control it. -/
def kernel (a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K _steps : Nat)
    (EVEN_K has_bias : Bool) : ComputeKernel :=
  sourceKernel a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr M N K
    stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
    BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K EVEN_K has_bias

def reuseIndicesKernel (a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K _steps : Nat)
    (EVEN_K has_bias : Bool) : ComputeKernel :=
  candidateKernel a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr M N K
    stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
    BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K EVEN_K has_bias

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.TypedSource
