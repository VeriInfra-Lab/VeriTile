import bench.tritonbench_g.rms_matmul_rbe.Kernels
import VeriTile.Triton
import bench.tritonbench_g.rbe_triton_transform.Proofs.Sources.Kernels
import VeriTile.Triton.DSL.IntegralJitFunctions

/-! Parameters for the literal-parameter RMS caller. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource
open VeriTile.Triton
structure Arguments where
  (X W RMS OUT : RegionName)
  (M N K sxb sxm sxk swk swn srms sob som son : Nat)
  (BM BN BK headDim : Nat)
  start : Int
  rotary : Bool
  (theta epsilon : ℝ)

end VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource

/-! The repaired RMS callee is shared verbatim with rms_matmul_rbe. This
interface realizes EPS/THETA through their original FP32 literal ports.
The separate rotary source and its helper below are complete Python bodies. -/
namespace VeriTile.Bench.Repairs.RmsRbeMatmul.TypedSource
open VeriTile.Triton
abbrev Arguments := VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource.Arguments
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

def source (fp8 : Bool) (p : Arguments) : ComputeKernel :=
  instantiate_source_procedure
    ({RmsMatmulRbeScalarDomain.TypedSource.rmsSource with materialized:=[]} : SourceProcedure)
    signature (RmsMatmulRbeScalarDomain.TypedSource.rmsSignature
      (if fp8 then .byte else .half) (if fp8 then .e5 else .half))
    procedure_args [x_ptr := p.X, w_ptr := p.W, rms_w_ptr := p.RMS, out_ptr := p.OUT, M := p.M, N := p.N, K := p.K, stride_x_batch := p.sxb, stride_x_m := p.sxm, stride_x_k := p.sxk, stride_w_k := p.swk, stride_w_n := p.swn, stride_rms_w := p.srms, stride_out_batch := p.sob, stride_out_m := p.som, stride_out_n := p.son, start_token_position := p.start, USE_FP8 := fp8, RBE_EPILOGUE := p.rotary, THETA := p.theta, EPS := p.epsilon, HEAD_DIM := p.headDim, BLOCK_SIZE_M := p.BM, BLOCK_SIZE_N := p.BN, BLOCK_SIZE_K := p.BK]

def indexCommutedKernel (fp8 : Bool) (p : Arguments) : ComputeKernel :=
  instantiate_source_procedure
    ({RmsMatmulRbeScalarDomain.TypedSource.rmsIndexCommutedSource with materialized:=[]} : SourceProcedure)
    signature (RmsMatmulRbeScalarDomain.TypedSource.rmsSignature
      (if fp8 then .byte else .half) (if fp8 then .e5 else .half))
    procedure_args [x_ptr := p.X, w_ptr := p.W, rms_w_ptr := p.RMS, out_ptr := p.OUT, M := p.M, N := p.N, K := p.K, stride_x_batch := p.sxb, stride_x_m := p.sxm, stride_x_k := p.sxk, stride_w_k := p.swk, stride_w_n := p.swn, stride_rms_w := p.srms, stride_out_batch := p.sob, stride_out_m := p.som, stride_out_n := p.son, start_token_position := p.start, USE_FP8 := fp8, RBE_EPILOGUE := p.rotary, THETA := p.theta, EPS := p.epsilon, HEAD_DIM := p.headDim, BLOCK_SIZE_M := p.BM, BLOCK_SIZE_N := p.BN, BLOCK_SIZE_K := p.BK]

def kernel (fp8 : Bool) (p : Arguments) : ComputeKernel := source fp8 p

/-- Complete original helper; DIM=128 is defined by its Python source. -/
def frequencySource : StatementJitFunction :=
  with_triton_params (offs_cn, starting_idx, theta, NB_TOKENS) constexpr_args [theta, NB_TOKENS] in triton {
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    freqs = offs_cn % DIM
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.pow(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS).to(tl.int64) + tl.cast(starting_idx, tl.int64))[:, None] / freqs[None, :]
    return tl.cos(freqs), tl.sin(freqs)
  }

def rbeSignature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("x_ptr",dtype),("out_ptr",dtype)]
  scalars := [("M",.integer),("K",.integer),("stride_x_batch",.integer),
    ("stride_x_m",.integer),("stride_x_n",.integer),("stride_out_batch",.integer),
    ("stride_out_m",.integer),("stride_out_n",.integer),("start_token_position",.integer)]
  constexpr := ["THETA","BLOCK_SIZE_M","BLOCK_SIZE_K"]
  readPorts := ["x_ptr"]
  writePorts := ["out_ptr"]

def rbeKernel (dtype : FloatDType) (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n : Nat)
    (start_token_position : Int) (THETA : ℝ) (BLOCK_SIZE_M BLOCK_SIZE_K : Nat) : ComputeKernel :=
  with_integral_operands in
  with_triton_call (rbeSignature dtype) call_args [M:=M,K:=K,
    stride_x_batch:=stride_x_batch,stride_x_m:=stride_x_m,stride_x_n:=stride_x_n,
    stride_out_batch:=stride_out_batch,stride_out_m:=stride_out_m,stride_out_n:=stride_out_n,
    start_token_position:=start_token_position] jit_functions [get_freq_multi_tokens:=frequencySource] in triton {
    M = tl.cast(M, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    start_token_position = tl.cast(start_token_position, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))
    pid_n = pid % (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64) * 2
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
            pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + stride_out_n, out_imag, mask=out_imag_mask)
  }

end
end VeriTile.Bench.Repairs.RmsRbeMatmul.TypedSource
