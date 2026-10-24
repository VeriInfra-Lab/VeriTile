import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPStages
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

/-! Typed copies of the nine previously accepted backward stage slices.
All original input .to(tl.float32) operations are retained. DHPrev/HPrev
are the explicit materialized carry boundary, not newly assumed results. -/

def dkHeadwise
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [DHPrev, q, do_, v, beta] [dk] (triton {
  tl.region DHPrev = tl.float32, q = tl.float32, do_ = tl.float32, v = tl.float32, beta = tl.float32, dk = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}).surfaceBody

def dkScalar
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [DHPrev, q, do_, v, beta] [dk] (triton {
  tl.region DHPrev = tl.float32, q = tl.float32, do_ = tl.float32, v = tl.float32, beta = tl.float32, dk = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_beta = tl.load(beta + i_bh * $(T) + $(t)).to(tl.float32)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}).surfaceBody

def dvHeadwise
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [DHPrev, q, do_, k, beta] [dv] (triton {
  tl.region DHPrev = tl.float32, q = tl.float32, do_ = tl.float32, k = tl.float32, beta = tl.float32, dv = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_v = d_v * b_beta
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}).surfaceBody

def dvScalar
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [DHPrev, q, do_, k, beta] [dv] (triton {
  tl.region DHPrev = tl.float32, q = tl.float32, do_ = tl.float32, k = tl.float32, beta = tl.float32, dv = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_beta = tl.load(beta + i_bh * $(T) + $(t)).to(tl.float32)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_v = d_v * b_beta
  tl.store(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), (d_v).to(dv.dtype.element_ty), mask=mask_bv)
}).surfaceBody

def dbetaHeadwise
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [DHPrev, q, do_, k, v] [dbeta] (triton {
  tl.region DHPrev = tl.float32, q = tl.float32, do_ = tl.float32, k = tl.float32, v = tl.float32, dbeta = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_beta = d_v * b_v
  tl.store(dbeta + (i_bh + i_k * $(B) * $(H) + i_v * $(B) * $(H) * $(NK)) * $(s_vo_h) +
    offs_v + $(t) * $(V), (d_beta).to(dbeta.dtype.element_ty), mask=mask_bv)
}).surfaceBody

def dbetaScalar
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [DHPrev, q, do_, k, v] [dbeta] (triton {
  tl.region DHPrev = tl.float32, q = tl.float32, do_ = tl.float32, k = tl.float32, v = tl.float32, dbeta = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(DHPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_q = tl.load(q + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32) * $(scale)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  d_h = prev + b_q[:, None] * b_do[None, :]
  d_v = tl.sum(d_h * b_k[:, None], axis=0)
  d_beta = tl.sum(d_v * b_v)
  tl.store(dbeta + (i_bh + i_v * $(B) * $(H)) * $(T) + $(t),
    (d_beta).to(dbeta.dtype.element_ty))
}).surfaceBody

def dqHeadwise
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [HPrev, k, v, beta, do_] [dq] (triton {
  tl.region HPrev = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, do_ = tl.float32, dq = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_beta = tl.load(beta + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_v *= b_beta
  h = prev + b_k[:, None] * b_v[None, :]
  _d_q = h * b_do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}).surfaceBody

def dqScalar
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    ComputeKernel :=
  ComputeKernel.mk [HPrev, k, v, beta, do_] [dq] (triton {
  tl.region HPrev = tl.float32, k = tl.float32, v = tl.float32, beta = tl.float32, do_ = tl.float32, dq = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  prev = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  b_k = tl.load(k + i_bh * $(s_qk_h) + i_k * $(BK) + offs_k + $(t) * $(K),
    mask=mask_bk, other=0.0).to(tl.float32)
  b_v = tl.load(v + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_do = tl.load(do_ + i_bh * $(s_vo_h) + i_v * $(BV) + offs_v + $(t) * $(V),
    mask=mask_bv, other=0.0).to(tl.float32)
  b_beta = tl.load(beta + i_bh * $(T) + $(t)).to(tl.float32)
  b_v *= b_beta
  h = prev + b_k[:, None] * b_v[None, :]
  _d_q = h * b_do[None, :]
  d_q = tl.sum(_d_q, axis=1) * $(scale)
  tl.store(dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_q).to(dq.dtype.element_ty), mask=mask_bk)
}).surfaceBody

def dkCorrection
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) :
    ComputeKernel :=
  ComputeKernel.mk [HPrev, dv, dk] [dk] (triton {
  tl.region HPrev = tl.float32, dv = tl.float32, dk = tl.float32
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  mask_bk = (i_k * $(BK) + offs_k) < $(K)
  mask_bv = (i_v * $(BV) + offs_v) < $(V)
  mask_kv = mask_bk[:, None] & mask_bv[None, :]
  d_k = tl.load(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), mask=mask_bk, other=0.0).to(tl.float32)
  d_v = tl.load(dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + offs_v +
    $(t) * $(V), mask=mask_bv, other=0.0).to(tl.float32)
  h = tl.load(HPrev + i_bh * $(K) * $(V) +
    (i_k * $(BK) + offs_k[:, None]) * $(V) + (i_v * $(BV) + offs_v[None, :]),
    mask=mask_kv, other=0.0)
  d_k -= tl.sum(d_v[None, :] * h, axis=1)
  tl.store(dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + offs_k +
    $(t) * $(K), (d_k).to(dk.dtype.element_ty), mask=mask_bk)
}).surfaceBody

end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
