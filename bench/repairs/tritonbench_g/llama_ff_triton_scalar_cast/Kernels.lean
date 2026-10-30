import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Source
open VeriTile.Triton Lean
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

/-- Readable shared DSL for both exact storage branches. Integer scalar casts,
all four K masks, each E5M2 conversion and the implicit half store are retained. -/
def half (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := triton {
  tl.ports inputs=[A, W1, W3, RMS], outputs=[OUT]
  tl.region A = tl.float16, W1 = tl.float16, W3 = tl.float16, RMS = tl.float16, OUT = tl.float16
  M64 = tl.cast($(M), tl.int64)
  N64 = tl.cast($(N), tl.int64)
  K64 = tl.cast($(K), tl.int64)
  sam64 = tl.cast($(sam), tl.int64)
  sak64 = tl.cast($(sak), tl.int64)
  sw1k64 = tl.cast($(sw1k), tl.int64)
  sw1n64 = tl.cast($(sw1n), tl.int64)
  sw3k64 = tl.cast($(sw3k), tl.int64)
  sw3n64 = tl.cast($(sw3n), tl.int64)
  soutm64 = tl.cast($(soutm), tl.int64)
  soutn64 = tl.cast($(soutn), tl.int64)
  srms64 = tl.cast($(srms), tl.int64)
  pid = tl.program_id(axis=0).to(tl.int64)
  pid_m = pid // ((N64 + $(BN) - $(1)) // $(BN))
  pid_n = pid % ((N64 + $(BN) - $(1)) // $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % M64
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % N64
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * sam64 + offs_k[None, :] * sak64)
  w1_ptrs = W1 + (offs_k[:, None] * sw1k64 + offs_bn[None, :] * sw1n64)
  w3_ptrs = W3 + (offs_k[:, None] * sw3k64 + offs_bn[None, :] * sw3n64)
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * srms64
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for step in range($(0), ((K64 + $(BK) - $(1)) // $(BK))) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    a = a * rms_w
    b = tl.load(w1_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * sak64
    w1_ptrs += $(BK) * sw1k64
    w3_ptrs += $(BK) * sw3k64
    rms_w_ptrs += $(BK) * srms64
  }
  a_mean = tl.sum(a_sum, axis=1) / K64 + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + (soutm64 * offs_outm[:, None] + soutn64 * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < M64) & (offs_outn[None, :] < N64)
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}

def byte (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := triton {
  tl.ports inputs=[A, W1, W3, RMS], outputs=[OUT]
  tl.region A = tl.float16, W1 = tl.int8, W3 = tl.int8, RMS = tl.int8, OUT = tl.float16
  M64 = tl.cast($(M), tl.int64)
  N64 = tl.cast($(N), tl.int64)
  K64 = tl.cast($(K), tl.int64)
  sam64 = tl.cast($(sam), tl.int64)
  sak64 = tl.cast($(sak), tl.int64)
  sw1k64 = tl.cast($(sw1k), tl.int64)
  sw1n64 = tl.cast($(sw1n), tl.int64)
  sw3k64 = tl.cast($(sw3k), tl.int64)
  sw3n64 = tl.cast($(sw3n), tl.int64)
  soutm64 = tl.cast($(soutm), tl.int64)
  soutn64 = tl.cast($(soutn), tl.int64)
  srms64 = tl.cast($(srms), tl.int64)
  pid = tl.program_id(axis=0).to(tl.int64)
  pid_m = pid // ((N64 + $(BN) - $(1)) // $(BN))
  pid_n = pid % ((N64 + $(BN) - $(1)) // $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % M64
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % N64
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * sam64 + offs_k[None, :] * sak64)
  w1_ptrs = W1 + (offs_k[:, None] * sw1k64 + offs_bn[None, :] * sw1n64)
  w3_ptrs = W3 + (offs_k[:, None] * sw3k64 + offs_bn[None, :] * sw3n64)
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * srms64
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for step in range($(0), ((K64 + $(BK) - $(1)) // $(BK))) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    rms_w = rms_w.to(tl.float8e5, bitcast=True)
    rms_w = rms_w.to(tl.float16)
    a = a * rms_w
    b = tl.load(w1_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    b = b.to(tl.float8e5, bitcast=True)
    b = b.to(tl.float32)
    b = b.to(tl.float16)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    c = c.to(tl.float8e5, bitcast=True)
    c = c.to(tl.float32)
    c = c.to(tl.float16)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * sak64
    w1_ptrs += $(BK) * sw1k64
    w3_ptrs += $(BK) * sw3k64
    rms_w_ptrs += $(BK) * srms64
  }
  a_mean = tl.sum(a_sum, axis=1) / K64 + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + (soutm64 * offs_outm[:, None] + soutn64 * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < M64) & (offs_outn[None, :] < N64)
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}

/-- Existing historical output-row index candidate, connected to the repaired
int64/four-mask source in both storage branches. No FP operation is reordered. -/
def halfCandidate (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := triton {
  tl.ports inputs=[A, W1, W3, RMS], outputs=[OUT]
  tl.region A = tl.float16, W1 = tl.float16, W3 = tl.float16, RMS = tl.float16, OUT = tl.float16
  M64 = tl.cast($(M), tl.int64)
  N64 = tl.cast($(N), tl.int64)
  K64 = tl.cast($(K), tl.int64)
  sam64 = tl.cast($(sam), tl.int64)
  sak64 = tl.cast($(sak), tl.int64)
  sw1k64 = tl.cast($(sw1k), tl.int64)
  sw1n64 = tl.cast($(sw1n), tl.int64)
  sw3k64 = tl.cast($(sw3k), tl.int64)
  sw3n64 = tl.cast($(sw3n), tl.int64)
  soutm64 = tl.cast($(soutm), tl.int64)
  soutn64 = tl.cast($(soutn), tl.int64)
  srms64 = tl.cast($(srms), tl.int64)
  pid = tl.program_id(axis=0).to(tl.int64)
  pid_m = pid // ((N64 + $(BN) - $(1)) // $(BN))
  pid_n = pid % ((N64 + $(BN) - $(1)) // $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % M64
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % N64
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * sam64 + offs_k[None, :] * sak64)
  w1_ptrs = W1 + (offs_k[:, None] * sw1k64 + offs_bn[None, :] * sw1n64)
  w3_ptrs = W3 + (offs_k[:, None] * sw3k64 + offs_bn[None, :] * sw3n64)
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * srms64
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for step in range($(0), ((K64 + $(BK) - $(1)) // $(BK))) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    a = a * rms_w
    b = tl.load(w1_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * sak64
    w1_ptrs += $(BK) * sw1k64
    w3_ptrs += $(BK) * sw3k64
    rms_w_ptrs += $(BK) * srms64
  }
  a_mean = tl.sum(a_sum, axis=1) / K64 + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = $(BM) * pid_m + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + (soutm64 * offs_outm[:, None] + soutn64 * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < M64) & (offs_outn[None, :] < N64)
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}

def byteCandidate (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat)
    (EPS : ℝ) : ComputeKernel := triton {
  tl.ports inputs=[A, W1, W3, RMS], outputs=[OUT]
  tl.region A = tl.float16, W1 = tl.int8, W3 = tl.int8, RMS = tl.int8, OUT = tl.float16
  M64 = tl.cast($(M), tl.int64)
  N64 = tl.cast($(N), tl.int64)
  K64 = tl.cast($(K), tl.int64)
  sam64 = tl.cast($(sam), tl.int64)
  sak64 = tl.cast($(sak), tl.int64)
  sw1k64 = tl.cast($(sw1k), tl.int64)
  sw1n64 = tl.cast($(sw1n), tl.int64)
  sw3k64 = tl.cast($(sw3k), tl.int64)
  sw3n64 = tl.cast($(sw3n), tl.int64)
  soutm64 = tl.cast($(soutm), tl.int64)
  soutn64 = tl.cast($(soutn), tl.int64)
  srms64 = tl.cast($(srms), tl.int64)
  pid = tl.program_id(axis=0).to(tl.int64)
  pid_m = pid // ((N64 + $(BN) - $(1)) // $(BN))
  pid_n = pid % ((N64 + $(BN) - $(1)) // $(BN))
  offs_am = (pid_m * $(BM) + tl.arange(0, $(BM))) % M64
  offs_bn = (pid_n * $(BN) + tl.arange(0, $(BN))) % N64
  offs_k = tl.arange(0, $(BK))
  a_ptrs = A + (offs_am[:, None] * sam64 + offs_k[None, :] * sak64)
  w1_ptrs = W1 + (offs_k[:, None] * sw1k64 + offs_bn[None, :] * sw1n64)
  w3_ptrs = W3 + (offs_k[:, None] * sw3k64 + offs_bn[None, :] * sw3n64)
  acc1 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  acc2 = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * srms64
  a_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for step in range($(0), ((K64 + $(BK) - $(1)) // $(BK))) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
    rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K64 - step * $(BK), other=0.0)
    rms_w = rms_w.to(tl.float8e5, bitcast=True)
    rms_w = rms_w.to(tl.float16)
    a = a * rms_w
    b = tl.load(w1_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    b = b.to(tl.float8e5, bitcast=True)
    b = b.to(tl.float32)
    b = b.to(tl.float16)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs, mask=offs_k[:, None] < K64 - step * $(BK), other=0.0)
    c = c.to(tl.float8e5, bitcast=True)
    c = c.to(tl.float32)
    c = c.to(tl.float16)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BK) * sak64
    w1_ptrs += $(BK) * sw1k64
    w3_ptrs += $(BK) * sw3k64
    rms_w_ptrs += $(BK) * srms64
  }
  a_mean = tl.sum(a_sum, axis=1) / K64 + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = $(BM) * pid_m + tl.arange(0, $(BM))
  offs_outn = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = OUT + (soutm64 * offs_outm[:, None] + soutn64 * offs_outn[None, :])
  out_mask = (offs_outm[:, None] < M64) & (offs_outn[None, :] < N64)
  tl.store(out_ptrs, accumulator.to(tl.float16), mask=out_mask)
}

def kernel (fp8 : Bool) (M N K : Nat) (EPS : ℝ) : ComputeKernel :=
  (if fp8 then byte else half) "A" "W1" "W3" "OUT" "RMS"
    M N K K 1 1 K 1 K N 1 1 16 16 64 EPS

/-- Repaired existing candidate on the exact selected launch profile. -/
def candidate (fp8 : Bool) (M N K : Nat) (EPS : ℝ) : ComputeKernel :=
  (if fp8 then byteCandidate else halfCandidate) "A" "W1" "W3" "OUT" "RMS"
    M N K K 1 1 K 1 K N 1 1 16 16 64 EPS

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Source
