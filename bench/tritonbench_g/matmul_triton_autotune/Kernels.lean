import VeriTile.Triton.DSL.ConditionalPureJitFunctions

/-! Canonical repaired source and a pure integer-index candidate. Caller ABI
metadata is outside the literal Python bodies. The source dynamic cdiv loop,
all signed casts, safe addresses, live dot carry and FP16 output are retained. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
open VeriTile.Triton
inductive Storage where | half | bfloat | single | e4m3 | e5m2
  deriving DecidableEq

def dtype : Storage → FloatDType | .half => .fp16 | .bfloat => .bf16 | .single => .fp32 | .e4m3 => .f8e4 | .e5m2 => .f8e5
structure Parameters where
  (A B C : RegionName)
  (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
  (ACTIVATION : Bool)

def Parameters.numKBlocks (p : Parameters) : Nat := (p.K+p.BLOCK_SIZE_K-1)/p.BLOCK_SIZE_K

end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Pasted
open VeriTile.Triton
noncomputable section

def callSignature (input_dtype : FloatDType) : CallSignature where
  pointers := [("c_ptr",.fp16)]
  floatingPointers := [("a_ptr",input_dtype),("b_ptr",input_dtype)]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "ACTIVATION"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def leakyReluSource : PureJitFunction := with_triton_params (x) constexpr_args [] in triton {
    """Applies the Leaky ReLU activation function."""
    return tl.where(x >= 0, x, 0.01 * x)
  }

def originalKernel (input_dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
    (ACTIVATION : String) : ComputeKernel :=
  with_triton_call (callSignature input_dtype) call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn]
    pure_functions [leaky_relu := leakyReluSource] in triton {
    """Kernel for computing the matrix multiplication C = A x B.
    A has shape (M, K), B has shape (K, N), and C has shape (M, N).
    """
    # Widen before every address/control product, including specialized strides.
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_kk = k * BLOCK_SIZE_K + offs_k
        # Inactive lanes form valid pointers as well; no final padded advance.
        safe_k = tl.where(offs_kk < K, offs_kk, 0)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=offs_kk[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_kk[:, None] < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    safe_cm = tl.where(offs_cm < M, offs_cm, 0)
    safe_cn = tl.where(offs_cn < N, offs_cn, 0)
    c_ptrs = c_ptr + stride_cm * safe_cm[:, None] + stride_cn * safe_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
  }
def candidateKernel (input_dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
    (ACTIVATION : String) : ComputeKernel :=
  with_triton_call (callSignature input_dtype) call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn]
    pure_functions [leaky_relu := leakyReluSource] in triton {
    """Kernel for computing the matrix multiplication C = A x B.
    A has shape (M, K), B has shape (K, N), and C has shape (M, N).
    """
    # Widen before every address/control product, including specialized strides.
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_n * GROUP_SIZE_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_kk = k * BLOCK_SIZE_K + offs_k
        # Inactive lanes form valid pointers as well; no final padded advance.
        safe_k = tl.where(offs_kk < K, offs_kk, 0)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=offs_kk[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_kk[:, None] < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    safe_cm = tl.where(offs_cm < M, offs_cm, 0)
    safe_cn = tl.where(offs_cn < N, offs_cn, 0)
    c_ptrs = c_ptr + stride_cm * safe_cm[:, None] + stride_cn * safe_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
  }
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Pasted

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
open VeriTile.Triton FPValue
noncomputable section

def typed (p : Parameters) (ACTIVATION : Bool) : ComputeKernel :=
  Pasted.candidateKernel .fp32 p.A p.B p.C p.M p.N p.K
    p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn
    p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M
    (if ACTIVATION then "leaky_relu" else "")
def kernel (p : Parameters) := typed p p.ACTIVATION
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
