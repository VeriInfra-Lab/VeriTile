import bench.repairs.tritonbench_g.rms_matmul_rbe.Kernels
import bench.repairs.tritonbench_g.rbe_triton_transform.Kernels

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.TypedSource
open VeriTile.Triton
abbrev Arguments := VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource.Arguments

/-- The complete repaired source, retaining FP32 X, the FP16 or signed E5
weight/RMS route, all masked K blocks, optional rotary, and typed output. -/
def source (fp8 : Bool) (p : Arguments) : ComputeKernel :=
  let X := p.X
  let W := p.W
  let RMS := p.RMS
  let OUT := p.OUT
  match fp8 with
  | Bool.false =>
    triton {
      tl.ports inputs=[X,W,RMS], outputs=[OUT]
      tl.region X = tl.float32, W = tl.float16, RMS = tl.float16, OUT = tl.float16
      pid_batch = tl.program_id(axis=0)
      pid = tl.program_id(axis=1)
      pid_m = pid // tl.cdiv($(p.N), $(p.BN))
      pid_n = pid % tl.cdiv($(p.N), $(p.BN))
      offs_m = (pid_m * $(p.BM) + tl.arange(0, $(p.BM))) % $(p.M)
      offs_n = (pid_n * $(p.BN) + tl.arange(0, $(p.BN))) % $(p.N)
      offs_k = tl.arange(0, $(p.BK))
      x_ptrs = X + pid_batch * $(p.sxb) + offs_m[:, None] * $(p.sxm) + offs_k[None, :] * $(p.sxk)
      w_ptrs = W + offs_k[:, None] * $(p.swk) + offs_n[None, :] * $(p.swn)
      accumulator = tl.zeros([$(p.BM), $(p.BN)], dtype=tl.float32)
      rms_w_ptrs = RMS + tl.arange(0, $(p.BK))[None, :] * $(p.srms)
      x_sum = tl.zeros([$(p.BM), $(p.BK)], dtype=tl.float32)
      for step in range($(0), tl.cdiv($(p.K), $(p.BK))) {
        x = tl.load(x_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < $(p.K) - step * $(p.BK), other=0.0)
        w = w.to(tl.float32)
        accumulator += tl.dot(x, w)
        x_ptrs += $(p.BK) * $(p.sxk)
        w_ptrs += $(p.BK) * $(p.swk)
        rms_w_ptrs += $(p.BK) * $(p.srms)
      }
      x_mean = tl.sum(x_sum, axis=1) / $(p.K) + $(p.epsilon)
      x_norm = tl.math.rsqrt(x_mean)
      accumulator = accumulator * x_norm[:, None]
      if $((p.rotary : Bool)) {
        pairs = tl.reshape(accumulator, [$(p.BM), $(p.BN / 2), $(2)])
        real = tl.split(pairs, 0)
        imag = tl.split(pairs, 1)
        pair_n = pid_n * $(p.BN) + $(2) * tl.arange(0, $(p.BN / 2))
        row_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
        exponent = tl.toReal(pair_n % $(p.headDim)).to(tl.float32) / $(p.headDim)
        frequency = tl.extra.cuda.libdevice.pow($(p.theta), exponent)
        position = $((p.start : Int)) + row_m
        angle = tl.toReal(position).to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), [$(p.BM), $(p.BN)])
      }
      offs_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
      offs_n = pid_n * $(p.BN) + tl.arange(0, $(p.BN))
      out_ptrs = OUT + pid_batch * $(p.sob) + offs_m[:, None] * $(p.som) + offs_n[None, :] * $(p.son)
      out_mask = (offs_m[:, None] < $(p.M)) & (offs_n[None, :] < $(p.N))
      tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
    }
  | Bool.true =>
    triton {
      tl.ports inputs=[X,W,RMS], outputs=[OUT]
      tl.region X = tl.float32, W = tl.int8, RMS = tl.int8, OUT = tl.float8e5
      pid_batch = tl.program_id(axis=0)
      pid = tl.program_id(axis=1)
      pid_m = pid // tl.cdiv($(p.N), $(p.BN))
      pid_n = pid % tl.cdiv($(p.N), $(p.BN))
      offs_m = (pid_m * $(p.BM) + tl.arange(0, $(p.BM))) % $(p.M)
      offs_n = (pid_n * $(p.BN) + tl.arange(0, $(p.BN))) % $(p.N)
      offs_k = tl.arange(0, $(p.BK))
      x_ptrs = X + pid_batch * $(p.sxb) + offs_m[:, None] * $(p.sxm) + offs_k[None, :] * $(p.sxk)
      w_ptrs = W + offs_k[:, None] * $(p.swk) + offs_n[None, :] * $(p.swn)
      accumulator = tl.zeros([$(p.BM), $(p.BN)], dtype=tl.float32)
      rms_w_ptrs = RMS + tl.arange(0, $(p.BK))[None, :] * $(p.srms)
      x_sum = tl.zeros([$(p.BM), $(p.BK)], dtype=tl.float32)
      for step in range($(0), tl.cdiv($(p.K), $(p.BK))) {
        x = tl.load(x_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        rms_w = rms_w.to(tl.float8e5, bitcast=True)
        rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < $(p.K) - step * $(p.BK), other=0.0)
        w = w.to(tl.float8e5, bitcast=True)
        w = w.to(tl.float32)
        w = w.to(tl.float16)
        w = w.to(tl.float32)
        accumulator += tl.dot(x, w)
        x_ptrs += $(p.BK) * $(p.sxk)
        w_ptrs += $(p.BK) * $(p.swk)
        rms_w_ptrs += $(p.BK) * $(p.srms)
      }
      x_mean = tl.sum(x_sum, axis=1) / $(p.K) + $(p.epsilon)
      x_norm = tl.math.rsqrt(x_mean)
      accumulator = accumulator * x_norm[:, None]
      if $((p.rotary : Bool)) {
        pairs = tl.reshape(accumulator, [$(p.BM), $(p.BN / 2), $(2)])
        real = tl.split(pairs, 0)
        imag = tl.split(pairs, 1)
        pair_n = pid_n * $(p.BN) + $(2) * tl.arange(0, $(p.BN / 2))
        row_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
        exponent = tl.toReal(pair_n % $(p.headDim)).to(tl.float32) / $(p.headDim)
        frequency = tl.extra.cuda.libdevice.pow($(p.theta), exponent)
        position = $((p.start : Int)) + row_m
        angle = tl.toReal(position).to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), [$(p.BM), $(p.BN)])
      }
      offs_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
      offs_n = pid_n * $(p.BN) + tl.arange(0, $(p.BN))
      out_ptrs = OUT + pid_batch * $(p.sob) + offs_m[:, None] * $(p.som) + offs_n[None, :] * $(p.son)
      out_mask = (offs_m[:, None] < $(p.M)) & (offs_n[None, :] < $(p.N))
      tl.store(out_ptrs, accumulator.to(tl.float8e5), mask=out_mask)
    }

/-- Historical alias retained for unchanged proof clients. The complete source
above is connected to it by SourcePair.readable_preserves. -/
def kernel (fp8 : Bool) (p : Arguments) :=
  VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource.kernel fp8 fp8 p

/-- Reuse the original-family output-row address candidate on the complete
repaired source. Only the final pid_m*BM becomes BM*pid_m. -/
def indexCommutedKernel (fp8 : Bool) (p : Arguments) : ComputeKernel :=
  let X := p.X
  let W := p.W
  let RMS := p.RMS
  let OUT := p.OUT
  match fp8 with
  | Bool.false =>
    triton {
      tl.ports inputs=[X,W,RMS], outputs=[OUT]
      tl.region X = tl.float32, W = tl.float16, RMS = tl.float16, OUT = tl.float16
      pid_batch = tl.program_id(axis=0)
      pid = tl.program_id(axis=1)
      pid_m = pid // tl.cdiv($(p.N), $(p.BN))
      pid_n = pid % tl.cdiv($(p.N), $(p.BN))
      offs_m = (pid_m * $(p.BM) + tl.arange(0, $(p.BM))) % $(p.M)
      offs_n = (pid_n * $(p.BN) + tl.arange(0, $(p.BN))) % $(p.N)
      offs_k = tl.arange(0, $(p.BK))
      x_ptrs = X + pid_batch * $(p.sxb) + offs_m[:, None] * $(p.sxm) + offs_k[None, :] * $(p.sxk)
      w_ptrs = W + offs_k[:, None] * $(p.swk) + offs_n[None, :] * $(p.swn)
      accumulator = tl.zeros([$(p.BM), $(p.BN)], dtype=tl.float32)
      rms_w_ptrs = RMS + tl.arange(0, $(p.BK))[None, :] * $(p.srms)
      x_sum = tl.zeros([$(p.BM), $(p.BK)], dtype=tl.float32)
      for step in range($(0), tl.cdiv($(p.K), $(p.BK))) {
        x = tl.load(x_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < $(p.K) - step * $(p.BK), other=0.0)
        w = w.to(tl.float32)
        accumulator += tl.dot(x, w)
        x_ptrs += $(p.BK) * $(p.sxk)
        w_ptrs += $(p.BK) * $(p.swk)
        rms_w_ptrs += $(p.BK) * $(p.srms)
      }
      x_mean = tl.sum(x_sum, axis=1) / $(p.K) + $(p.epsilon)
      x_norm = tl.math.rsqrt(x_mean)
      accumulator = accumulator * x_norm[:, None]
      if $((p.rotary : Bool)) {
        pairs = tl.reshape(accumulator, [$(p.BM), $(p.BN / 2), $(2)])
        real = tl.split(pairs, 0)
        imag = tl.split(pairs, 1)
        pair_n = pid_n * $(p.BN) + $(2) * tl.arange(0, $(p.BN / 2))
        row_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
        exponent = tl.toReal(pair_n % $(p.headDim)).to(tl.float32) / $(p.headDim)
        frequency = tl.extra.cuda.libdevice.pow($(p.theta), exponent)
        position = $((p.start : Int)) + row_m
        angle = tl.toReal(position).to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), [$(p.BM), $(p.BN)])
      }
      offs_m = $(p.BM) * pid_m + tl.arange(0, $(p.BM))
      offs_n = pid_n * $(p.BN) + tl.arange(0, $(p.BN))
      out_ptrs = OUT + pid_batch * $(p.sob) + offs_m[:, None] * $(p.som) + offs_n[None, :] * $(p.son)
      out_mask = (offs_m[:, None] < $(p.M)) & (offs_n[None, :] < $(p.N))
      tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
    }
  | Bool.true =>
    triton {
      tl.ports inputs=[X,W,RMS], outputs=[OUT]
      tl.region X = tl.float32, W = tl.int8, RMS = tl.int8, OUT = tl.float8e5
      pid_batch = tl.program_id(axis=0)
      pid = tl.program_id(axis=1)
      pid_m = pid // tl.cdiv($(p.N), $(p.BN))
      pid_n = pid % tl.cdiv($(p.N), $(p.BN))
      offs_m = (pid_m * $(p.BM) + tl.arange(0, $(p.BM))) % $(p.M)
      offs_n = (pid_n * $(p.BN) + tl.arange(0, $(p.BN))) % $(p.N)
      offs_k = tl.arange(0, $(p.BK))
      x_ptrs = X + pid_batch * $(p.sxb) + offs_m[:, None] * $(p.sxm) + offs_k[None, :] * $(p.sxk)
      w_ptrs = W + offs_k[:, None] * $(p.swk) + offs_n[None, :] * $(p.swn)
      accumulator = tl.zeros([$(p.BM), $(p.BN)], dtype=tl.float32)
      rms_w_ptrs = RMS + tl.arange(0, $(p.BK))[None, :] * $(p.srms)
      x_sum = tl.zeros([$(p.BM), $(p.BK)], dtype=tl.float32)
      for step in range($(0), tl.cdiv($(p.K), $(p.BK))) {
        x = tl.load(x_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < $(p.K) - step * $(p.BK), other=0.0)
        rms_w = rms_w.to(tl.float8e5, bitcast=True)
        rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < $(p.K) - step * $(p.BK), other=0.0)
        w = w.to(tl.float8e5, bitcast=True)
        w = w.to(tl.float32)
        w = w.to(tl.float16)
        w = w.to(tl.float32)
        accumulator += tl.dot(x, w)
        x_ptrs += $(p.BK) * $(p.sxk)
        w_ptrs += $(p.BK) * $(p.swk)
        rms_w_ptrs += $(p.BK) * $(p.srms)
      }
      x_mean = tl.sum(x_sum, axis=1) / $(p.K) + $(p.epsilon)
      x_norm = tl.math.rsqrt(x_mean)
      accumulator = accumulator * x_norm[:, None]
      if $((p.rotary : Bool)) {
        pairs = tl.reshape(accumulator, [$(p.BM), $(p.BN / 2), $(2)])
        real = tl.split(pairs, 0)
        imag = tl.split(pairs, 1)
        pair_n = pid_n * $(p.BN) + $(2) * tl.arange(0, $(p.BN / 2))
        row_m = pid_m * $(p.BM) + tl.arange(0, $(p.BM))
        exponent = tl.toReal(pair_n % $(p.headDim)).to(tl.float32) / $(p.headDim)
        frequency = tl.extra.cuda.libdevice.pow($(p.theta), exponent)
        position = $((p.start : Int)) + row_m
        angle = tl.toReal(position).to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), [$(p.BM), $(p.BN)])
      }
      offs_m = $(p.BM) * pid_m + tl.arange(0, $(p.BM))
      offs_n = pid_n * $(p.BN) + tl.arange(0, $(p.BN))
      out_ptrs = OUT + pid_batch * $(p.sob) + offs_m[:, None] * $(p.som) + offs_n[None, :] * $(p.son)
      out_mask = (offs_m[:, None] < $(p.M)) & (offs_n[None, :] < $(p.N))
      tl.store(out_ptrs, accumulator.to(tl.float8e5), mask=out_mask)
    }

abbrev rawRotary := VeriTile.Bench.Repairs.RbeTritonTransform.kernel
end VeriTile.Bench.Repairs.RmsRbeMatmul.TypedSource
