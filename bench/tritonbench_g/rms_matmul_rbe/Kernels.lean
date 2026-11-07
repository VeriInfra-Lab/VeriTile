import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.SourceProcedures

/-! Complete repaired Python callees, including every integer cast, precision
branch and implicit store conversion. Pointer storage comes from the launch;
EPS/THETA are immutable constexpr values, symbolically materialized as FP32
by the existing scalar IO interface. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.TypedSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Arguments where
  (X W RMS OUT : RegionName)
  (M N K sxb sxm sxk swk swn srms sob som son : Nat)
  (BM BN BK headDim : Nat)
  start : Int
  rotary : Bool

inductive InputStorage where | half | byte
inductive OutputStorage where | half | e5

def rmsSignature (input : InputStorage) (output : OutputStorage) : CallSignature where
  pointers := [("x_ptr",.fp32),
    ("w_ptr",match input with | .half => .fp16 | .byte => .i8),
    ("rms_w_ptr",match input with | .half => .fp16 | .byte => .i8),
    ("out_ptr",match output with | .half => .fp16 | .e5 => .f8e5)]
  scalars := [("M",.integer),("N",.integer),("K",.integer),
    ("stride_x_batch",.integer),("stride_x_m",.integer),("stride_x_k",.integer),
    ("stride_w_k",.integer),("stride_w_n",.integer),("stride_rms_w",.integer),
    ("stride_out_batch",.integer),("stride_out_m",.integer),("stride_out_n",.integer),
    ("start_token_position",.integer)]
  constexpr := ["USE_FP8","RBE_EPILOGUE","THETA","EPS","HEAD_DIM",
    "BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K"]

def qkvSignature (input : InputStorage) (output : OutputStorage) : CallSignature where
  pointers := [("x_ptr",.fp32),
    ("q_weight_ptr",(match input with | .half => .fp16 | .byte => .i8)),
    ("k_weight_ptr",(match input with | .half => .fp16 | .byte => .i8)),
    ("v_weight_ptr",(match input with | .half => .fp16 | .byte => .i8)),
    ("rms_w_ptr",(match input with | .half => .fp16 | .byte => .i8)),
    ("q_ptr",.fp16),
    ("k_ptr",(match output with | .half => .fp16 | .e5 => .f8e5)),
    ("v_ptr",(match output with | .half => .fp16 | .e5 => .f8e5))]
  scalars := [("M",.integer), ("N",.integer), ("K",.integer), ("stride_x_batch",.integer), ("stride_x_m",.integer), ("stride_x_k",.integer), ("stride_q_w_k",.integer), ("stride_q_w_n",.integer), ("stride_k_w_k",.integer), ("stride_k_w_n",.integer), ("stride_v_w_k",.integer), ("stride_v_w_n",.integer), ("stride_rms_w",.integer), ("stride_q_batch",.integer), ("stride_q_m",.integer), ("stride_q_n",.integer), ("stride_k_batch",.integer), ("stride_k_m",.integer), ("stride_k_n",.integer), ("stride_v_batch",.integer), ("stride_v_m",.integer), ("stride_v_n",.integer), ("start_token_position",.integer)]
  constexpr := ["USE_FP8", "THETA", "EPS", "HEAD_DIM", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]
  readPorts := ["x_ptr","q_weight_ptr","k_weight_ptr","v_weight_ptr","rms_w_ptr"]
  writePorts := ["q_ptr","k_ptr","v_ptr"]

def rmsSource : SourceProcedure :=
  with_triton_params (x_ptr, w_ptr, rms_w_ptr, out_ptr, M, N, K, stride_x_batch, stride_x_m, stride_x_k, stride_w_k, stride_w_n, stride_rms_w, stride_out_batch, stride_out_m, stride_out_n, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    constexpr_args [USE_FP8, RBE_EPILOGUE, THETA, EPS, HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
    fp32_constants [THETA, EPS] in triton {

    """
    Prologue: RMS
    Epilogue: nothing or Rotary embeddings
    c = ROBE((rms(a) * rms_w) @ b)
    """
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_k = tl.cast(stride_x_k, tl.int64)
    stride_w_k = tl.cast(stride_w_k, tl.int64)
    stride_w_n = tl.cast(stride_w_n, tl.int64)
    stride_rms_w = tl.cast(stride_rms_w, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))
    pid_n = pid % (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)[None, :] * stride_rms_w
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for step in range(0, (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        w = w.to(x.dtype)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    # Rotate normalized FP32 pairs before the source's sole output cast.
    if RBE_EPILOGUE:
        real, imag = tl.split(tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)))
        pair_n = pid_n * BLOCK_SIZE_N + 2 * tl.arange(0, BLOCK_SIZE_N // 2).to(tl.int64)
        row_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        exponent = (pair_n % HEAD_DIM).to(tl.float32) / HEAD_DIM
        frequency = tl.extra.cuda.libdevice.pow(THETA, exponent)
        position = tl.cast(start_token_position, tl.int64) + row_m
        angle = position.to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)

  }

def rmsKernel (input : InputStorage) (output : OutputStorage)
    (x_ptr w_ptr rms_w_ptr out_ptr : RegionName)
    (M N K stride_x_batch stride_x_m stride_x_k stride_w_k stride_w_n stride_rms_w
      stride_out_batch stride_out_m stride_out_n : Nat)
    (start_token_position : Int) (RBE_EPILOGUE : Bool)
    (HEAD_DIM BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : ComputeKernel :=
  let USE_FP8 : Bool := match input with | .half => Bool.false | .byte => Bool.true
  instantiate_source_procedure rmsSource signature (rmsSignature input output)
    procedure_args [x_ptr := x_ptr, w_ptr := w_ptr, rms_w_ptr := rms_w_ptr, out_ptr := out_ptr, M := M, N := N, K := K, stride_x_batch := stride_x_batch, stride_x_m := stride_x_m, stride_x_k := stride_x_k, stride_w_k := stride_w_k, stride_w_n := stride_w_n, stride_rms_w := stride_rms_w, stride_out_batch := stride_out_batch, stride_out_m := stride_out_m, stride_out_n := stride_out_n, start_token_position := start_token_position, USE_FP8 := USE_FP8, RBE_EPILOGUE := RBE_EPILOGUE, HEAD_DIM := HEAD_DIM, BLOCK_SIZE_M := BLOCK_SIZE_M, BLOCK_SIZE_N := BLOCK_SIZE_N, BLOCK_SIZE_K := BLOCK_SIZE_K]

def rmsIndexCommutedSource : SourceProcedure :=
  with_triton_params (x_ptr, w_ptr, rms_w_ptr, out_ptr, M, N, K, stride_x_batch, stride_x_m, stride_x_k, stride_w_k, stride_w_n, stride_rms_w, stride_out_batch, stride_out_m, stride_out_n, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    constexpr_args [USE_FP8, RBE_EPILOGUE, THETA, EPS, HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
    fp32_constants [THETA, EPS] in triton {

    """
    Prologue: RMS
    Epilogue: nothing or Rotary embeddings
    c = ROBE((rms(a) * rms_w) @ b)
    """
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_k = tl.cast(stride_x_k, tl.int64)
    stride_w_k = tl.cast(stride_w_k, tl.int64)
    stride_w_n = tl.cast(stride_w_n, tl.int64)
    stride_rms_w = tl.cast(stride_rms_w, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))
    pid_n = pid % (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)[None, :] * stride_rms_w
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for step in range(0, (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        w = w.to(x.dtype)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    # Rotate normalized FP32 pairs before the source's sole output cast.
    if RBE_EPILOGUE:
        real, imag = tl.split(tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)))
        pair_n = pid_n * BLOCK_SIZE_N + 2 * tl.arange(0, BLOCK_SIZE_N // 2).to(tl.int64)
        row_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        exponent = (pair_n % HEAD_DIM).to(tl.float32) / HEAD_DIM
        frequency = tl.extra.cuda.libdevice.pow(THETA, exponent)
        position = tl.cast(start_token_position, tl.int64) + row_m
        angle = position.to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)

  }

def rmsIndexCommutedKernel (input : InputStorage) (output : OutputStorage)
    (x_ptr w_ptr rms_w_ptr out_ptr : RegionName)
    (M N K stride_x_batch stride_x_m stride_x_k stride_w_k stride_w_n stride_rms_w
      stride_out_batch stride_out_m stride_out_n : Nat)
    (start_token_position : Int) (RBE_EPILOGUE : Bool)
    (HEAD_DIM BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : ComputeKernel :=
  let USE_FP8 : Bool := match input with | .half => Bool.false | .byte => Bool.true
  instantiate_source_procedure rmsIndexCommutedSource signature (rmsSignature input output)
    procedure_args [x_ptr := x_ptr, w_ptr := w_ptr, rms_w_ptr := rms_w_ptr, out_ptr := out_ptr, M := M, N := N, K := K, stride_x_batch := stride_x_batch, stride_x_m := stride_x_m, stride_x_k := stride_x_k, stride_w_k := stride_w_k, stride_w_n := stride_w_n, stride_rms_w := stride_rms_w, stride_out_batch := stride_out_batch, stride_out_m := stride_out_m, stride_out_n := stride_out_n, start_token_position := start_token_position, USE_FP8 := USE_FP8, RBE_EPILOGUE := RBE_EPILOGUE, HEAD_DIM := HEAD_DIM, BLOCK_SIZE_M := BLOCK_SIZE_M, BLOCK_SIZE_N := BLOCK_SIZE_N, BLOCK_SIZE_K := BLOCK_SIZE_K]

def halfHalf (p : Arguments) : ComputeKernel :=
  rmsKernel .half .half p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def halfFP8 (p : Arguments) : ComputeKernel :=
  rmsKernel .half .e5 p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def fp8Half (p : Arguments) : ComputeKernel :=
  rmsKernel .byte .half p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def fp8FP8 (p : Arguments) : ComputeKernel :=
  rmsKernel .byte .e5 p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def halfHalfIndexCommuted (p : Arguments) : ComputeKernel :=
  rmsIndexCommutedKernel .half .half p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def halfFP8IndexCommuted (p : Arguments) : ComputeKernel :=
  rmsIndexCommutedKernel .half .e5 p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def fp8HalfIndexCommuted (p : Arguments) : ComputeKernel :=
  rmsIndexCommutedKernel .byte .half p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def fp8FP8IndexCommuted (p : Arguments) : ComputeKernel :=
  rmsIndexCommutedKernel .byte .e5 p.X p.W p.RMS p.OUT p.M p.N p.K
    p.sxb p.sxm p.sxk p.swk p.swn p.srms p.sob p.som p.son
    p.start p.rotary p.headDim p.BM p.BN p.BK

def kernel (fp8 outputFP8 : Bool) : Arguments → ComputeKernel :=
  match fp8,outputFP8 with
  | Bool.false,Bool.false => halfHalf
  | Bool.false,Bool.true => halfFP8
  | Bool.true,Bool.false => fp8Half
  | Bool.true,Bool.true => fp8FP8

def indexCommuted (fp8 outputFP8 : Bool) : Arguments → ComputeKernel :=
  match fp8,outputFP8 with
  | Bool.false,Bool.false => halfHalfIndexCommuted
  | Bool.false,Bool.true => halfFP8IndexCommuted
  | Bool.true,Bool.false => fp8HalfIndexCommuted
  | Bool.true,Bool.true => fp8FP8IndexCommuted

def qkvIndexCommuted (fp8 : Bool) (q k v : Arguments) : ComputeKernel :=
  .mk [q.X,q.W,k.W,v.W,q.RMS] [q.OUT,k.OUT,v.OUT]
    ((indexCommuted fp8 Bool.false {q with rotary:=Bool.true}).surfaceBody ++
     (indexCommuted fp8 fp8 {k with rotary:=Bool.true}).surfaceBody ++
     (indexCommuted fp8 fp8 {v with rotary:=Bool.false}).surfaceBody)

def qkvKernel (input : InputStorage) (output : OutputStorage)
    (x_ptr q_weight_ptr k_weight_ptr v_weight_ptr rms_w_ptr q_ptr k_ptr v_ptr : RegionName)
    (M N K stride_x_batch stride_x_m stride_x_k stride_q_w_k stride_q_w_n stride_k_w_k stride_k_w_n stride_v_w_k stride_v_w_n stride_rms_w stride_q_batch stride_q_m stride_q_n stride_k_batch stride_k_m stride_k_n stride_v_batch stride_v_m stride_v_n : Nat)
    (start_token_position : Int) (HEAD_DIM BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : ComputeKernel :=
  let USE_FP8 : Bool := match input with | .half => Bool.false | .byte => Bool.true
  with_triton_call (qkvSignature input output) call_args [M := M, N := N, K := K, stride_x_batch := stride_x_batch, stride_x_m := stride_x_m, stride_x_k := stride_x_k, stride_q_w_k := stride_q_w_k, stride_q_w_n := stride_q_w_n, stride_k_w_k := stride_k_w_k, stride_k_w_n := stride_k_w_n, stride_v_w_k := stride_v_w_k, stride_v_w_n := stride_v_w_n, stride_rms_w := stride_rms_w, stride_q_batch := stride_q_batch, stride_q_m := stride_q_m, stride_q_n := stride_q_n, stride_k_batch := stride_k_batch, stride_k_m := stride_k_m, stride_k_n := stride_k_n, stride_v_batch := stride_v_batch, stride_v_m := stride_v_m, stride_v_n := stride_v_n, start_token_position := start_token_position]
    fp32_constants [THETA, EPS] source_procedures [rms_matmul_rbe := rmsSource] in triton {
    # q
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=q_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=q_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_q_w_k, stride_w_n=stride_q_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_q_batch, stride_out_m=stride_q_m, stride_out_n=stride_q_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=True,
        THETA=THETA,
        EPS=EPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # k
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=k_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=k_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_k_w_k, stride_w_n=stride_k_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_k_batch, stride_out_m=stride_k_m, stride_out_n=stride_k_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=True,
        THETA=THETA,
        EPS=EPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # v
    rms_matmul_rbe(
        x_ptr=x_ptr,
        w_ptr=v_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=v_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_v_w_k, stride_w_n=stride_v_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_v_batch, stride_out_m=stride_v_m, stride_out_n=stride_v_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=False,
        THETA=THETA,
        EPS=EPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
  }

def qkvIndexCommutedKernel (input : InputStorage) (output : OutputStorage)
    (x_ptr q_weight_ptr k_weight_ptr v_weight_ptr rms_w_ptr q_ptr k_ptr v_ptr : RegionName)
    (M N K stride_x_batch stride_x_m stride_x_k stride_q_w_k stride_q_w_n stride_k_w_k stride_k_w_n stride_v_w_k stride_v_w_n stride_rms_w stride_q_batch stride_q_m stride_q_n stride_k_batch stride_k_m stride_k_n stride_v_batch stride_v_m stride_v_n : Nat)
    (start_token_position : Int) (HEAD_DIM BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : ComputeKernel :=
  let USE_FP8 : Bool := match input with | .half => Bool.false | .byte => Bool.true
  with_triton_call (qkvSignature input output) call_args [M := M, N := N, K := K, stride_x_batch := stride_x_batch, stride_x_m := stride_x_m, stride_x_k := stride_x_k, stride_q_w_k := stride_q_w_k, stride_q_w_n := stride_q_w_n, stride_k_w_k := stride_k_w_k, stride_k_w_n := stride_k_w_n, stride_v_w_k := stride_v_w_k, stride_v_w_n := stride_v_w_n, stride_rms_w := stride_rms_w, stride_q_batch := stride_q_batch, stride_q_m := stride_q_m, stride_q_n := stride_q_n, stride_k_batch := stride_k_batch, stride_k_m := stride_k_m, stride_k_n := stride_k_n, stride_v_batch := stride_v_batch, stride_v_m := stride_v_m, stride_v_n := stride_v_n, start_token_position := start_token_position]
    fp32_constants [THETA, EPS] source_procedures [rms_matmul_rbe_index_commuted := rmsIndexCommutedSource] in triton {
    # q
    rms_matmul_rbe_index_commuted(
        x_ptr=x_ptr,
        w_ptr=q_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=q_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_q_w_k, stride_w_n=stride_q_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_q_batch, stride_out_m=stride_q_m, stride_out_n=stride_q_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=True,
        THETA=THETA,
        EPS=EPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # k
    rms_matmul_rbe_index_commuted(
        x_ptr=x_ptr,
        w_ptr=k_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=k_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_k_w_k, stride_w_n=stride_k_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_k_batch, stride_out_m=stride_k_m, stride_out_n=stride_k_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=True,
        THETA=THETA,
        EPS=EPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    # v
    rms_matmul_rbe_index_commuted(
        x_ptr=x_ptr,
        w_ptr=v_weight_ptr, rms_w_ptr=rms_w_ptr, out_ptr=v_ptr,
        M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=stride_x_k,
        stride_w_k=stride_v_w_k, stride_w_n=stride_v_w_n,
        stride_rms_w=stride_rms_w,
        stride_out_batch=stride_v_batch, stride_out_m=stride_v_m, stride_out_n=stride_v_n,
        start_token_position=start_token_position,
        USE_FP8=USE_FP8,
        RBE_EPILOGUE=False,
        THETA=THETA,
        EPS=EPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
  }

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.TypedSource
