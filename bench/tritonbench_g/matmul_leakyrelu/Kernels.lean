import VeriTile.Triton.DSL.ConditionalPureJitFunctions
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPValue
open VeriTile.Triton
inductive Storage where | half | bfloat | single | e4m3 | e5m2
  deriving DecidableEq

def dtype : Storage → FloatDType | .half => .fp16 | .bfloat => .bf16 | .single => .fp32 | .e4m3 => .f8e4 | .e5m2 => .f8e5
structure Parameters where
  (A B C : RegionName)
  (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
  (ACTIVATION : Bool)
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPValue

namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
open VeriTile.Triton
inductive Storage where | half | bfloat | single | e4m3 | e5m2
  deriving DecidableEq

def dtype : Storage → FloatDType | .half => .fp16 | .bfloat => .bf16 | .single => .fp32 | .e4m3 => .f8e4 | .e5m2 => .f8e5
structure Parameters where
  (A B C : RegionName)
  (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat)
  (ACTIVATION : Bool)
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused

namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Pasted
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000

def callSignature (input_dtype : FloatDType) : CallSignature where
  pointers := [("c_ptr",.fp16)]
  floatingPointers := [("a_ptr",input_dtype),("b_ptr",input_dtype)]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("stride_am",.integer), ("stride_ak",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_cm",.integer), ("stride_cn",.integer)]
  constexpr := ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "ACTIVATION"]
  readPorts := ["a_ptr","b_ptr"]
  writePorts := ["c_ptr"]

def leakyReluSource : PureJitFunction := with_triton_params (x) constexpr_args [] in triton {
    return tl.where(x >= 0, x, 0.01 * x)
  }

def leakyReluRhsSource : PureJitFunction := with_triton_params (x) constexpr_args [] in triton {
    return tl.where(x >= 0, x, x * 0.01)
  }

def originalKernel (input_dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) (ACTIVATION : String) : ComputeKernel :=
  with_triton_call (callSignature input_dtype) call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn]
    pure_functions [leaky_relu := leakyReluSource] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function if specified
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
  }

def candidateKernel (input_dtype : FloatDType) (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) (ACTIVATION : String) : ComputeKernel :=
  with_triton_call (callSignature input_dtype) call_args [M := M, N := N, K := K, stride_am := stride_am, stride_ak := stride_ak, stride_bk := stride_bk, stride_bn := stride_bn, stride_cm := stride_cm, stride_cn := stride_cn]
    pure_functions [leaky_relu_rhs := leakyReluRhsSource] in triton {
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function if specified
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu_rhs(accumulator)

    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
  }

end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Pasted
