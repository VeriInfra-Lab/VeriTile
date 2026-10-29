import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

/-- The four entire previously accepted semantic stages. DTYPE explicitly
records physical storage; fp32 register conversions materialize the original
Python .to(tl.float32) and staged fp32 register boundaries. -/
def kvTyped (dtype : FloatDType)
    (KVPrev KTrans V KVOut : RegionName) (D BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  tl.region KVPrev = DTYPE, KTrans = DTYPE, V = DTYPE, KVOut = DTYPE
  offs_a = tl.arange(0, $(D))
  offs_j = tl.arange(0, $(BLOCK))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  prev = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :]).to(tl.float32)
  k_trans = tl.load(KTrans + offs_a[:, None] * $(BLOCK) + offs_j[None, :]).to(tl.float32)
  v = tl.load(V + offs_j[:, None] * $(BLOCK_MODEL) + offs_c[None, :]).to(tl.float32)
  kv_update = tl.dot(k_trans, v)
  kv = prev + kv_update
  tl.store(KVOut + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :], kv.to(DTYPE))
}

def interTyped (dtype : FloatDType)
    (Q KVPrev OInter : RegionName) (BLOCK D BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  tl.region Q = DTYPE, KVPrev = DTYPE, OInter = DTYPE
  offs_r = tl.arange(0, $(BLOCK))
  offs_a = tl.arange(0, $(D))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  q = tl.load(Q + offs_r[:, None] * $(D) + offs_a[None, :]).to(tl.float32)
  kv = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :]).to(tl.float32)
  o_inter = tl.dot(q, kv)
  tl.store(OInter + offs_r[:, None] * $(BLOCK_MODEL) + offs_c[None, :], o_inter.to(DTYPE))
}

def bodyTyped (dtype : FloatDType)
    (Q K V KVPrev Out : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat) : ComputeKernel := triton {
  tl.region Q = DTYPE, K = DTYPE, V = DTYPE, KVPrev = DTYPE, Out = DTYPE
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  e_offset = off_e * $(BLOCK_MODEL)
  offs_a = tl.arange(0, $(d))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  offs_r = tl.arange(0, $(BLOCK))
  off_block = $(m) * $(BLOCK) + tl.arange(0, $(BLOCK))
  q = tl.load(Q + qk_offset + off_block[:, None] * $(d) + offs_a[None, :],
    mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
  k_trans = tl.load(K + qk_offset + off_block[None, :] * $(d) + offs_a[:, None],
    mask=off_block[None, :] < $(n), other=0.0).to(tl.float32)
  v = tl.load(V + v_offset + off_block[:, None] * $(e) + e_offset + offs_c[None, :],
    mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
  kv = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :]).to(tl.float32)
  qk = tl.dot(q, k_trans)
  qk = tl.where(offs_r[:, None] >= offs_r[None, :], qk, 0.0)
  o_intra = tl.dot(qk, v)
  o_inter = tl.dot(q, kv)
  o = o_intra + o_inter
  tl.store(Out + o_offset + off_block[:, None] * $(e) + e_offset + offs_c[None, :],
    (o).to(Out.dtype.element_ty), mask=off_block[:, None] < $(n))
}

def gradTyped (dtype : FloatDType)
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  tl.region GradPre = DTYPE, Out = DTYPE
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  grad = tl.load(GradPre + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (grad).to(Out.dtype.element_ty), mask=mask)
}

structure Parameters where
  (query key value carry output : RegionName)
  (blockIndex length keyDim valueDim rows columns : Nat)

def kvSource (p : Parameters) := kvTyped .fp32 p.carry p.key p.value p.output p.keyDim p.rows p.columns
def interSource (p : Parameters) := interTyped .fp32 p.query p.carry p.output p.rows p.keyDim p.columns
def bodySource (p : Parameters) := bodyTyped .fp32 p.query p.key p.value p.carry p.output
  p.blockIndex p.length p.keyDim p.valueDim p.rows p.columns
def gradSource (p : Parameters) := gradTyped .fp32 p.query p.output p.length p.valueDim p.rows p.columns
end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
