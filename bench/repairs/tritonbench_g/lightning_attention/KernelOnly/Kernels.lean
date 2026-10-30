import bench.repairs.tritonbench_g.lightning_attention.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile.Triton

def forward (p : Parameters) : ComputeKernel :=
  let b := p.B
  let h := p.H
  let n := p.N
  let d := p.D
  let e := p.E
  let BLOCK := p.BT
  let NUM_BLOCK := p.NT
  let BLOCK_MODEL := p.BM
  let CBLOCK := p.CB
  let NUM_CBLOCK := p.NC
  let Q : RegionName := "query"
  let K : RegionName := "key"
  let V : RegionName := "value"
  let Out : RegionName := "output"
  triton {
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32
    off_bh = tl.program_id(0)
    off_bh % $(h)
    off_e = tl.program_id(1)
    qk_offset = off_bh * $(n) * $(d)
    v_offset = off_bh * $(n) * $(e)
    o_offset = off_bh * $(n) * $(e)
    e_offset = off_e * $(BLOCK_MODEL)
    Q_block_ptr = Q + qk_offset + tl.arange(0, $(d))[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, $(d))[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
    off_block = tl.arange(0, $(BLOCK))
    index = off_block[:, None].to(tl.int64) - off_block[None, :].to(tl.int64)
    kv = tl.zeros([$(d), $(BLOCK_MODEL)], dtype=tl.float32)
    for i in range($(NUM_BLOCK)) {
      q = tl.load(Q_block_ptr + off_block[:, None] * $(d), mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
      k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * $(d), mask=off_block[None, :] < $(n), other=0.0).to(tl.float32)
      v = tl.load(V_block_ptr + off_block[:, None] * $(e), mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
      qk = tl.dot(q, k_trans)
      qk = tl.where(index >= $((0 : Int)), qk, 0)
      o_intra = tl.dot(qk, v)
      o_inter = tl.dot(q, kv)
      o = o_intra + o_inter
      tl.store(O_block_ptr + off_block[:, None] * $(e), o.to(tl.float32), mask=off_block[:, None] < $(n))
      kv += tl.dot(k_trans, v)
      off_block += $(BLOCK)
    }
  }

def intra (p : Parameters) : ComputeKernel :=
  let b := p.B
  let h := p.H
  let n := p.N
  let d := p.D
  let e := p.E
  let BLOCK := p.BT
  let NUM_BLOCK := p.NT
  let BLOCK_MODEL := p.BM
  let CBLOCK := p.CB
  let NUM_CBLOCK := p.NC
  let Q : RegionName := "query"
  let K : RegionName := "key"
  let V : RegionName := "value"
  let DO : RegionName := "incoming"
  let DQ : RegionName := "queryGradient"
  let DK : RegionName := "keyGradient"
  let DV : RegionName := "valueGradient"
  triton {
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32, DO = tl.float32, DQ = tl.float32, DK = tl.float32, DV = tl.float32
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_bh % $(h)
    qk_offset = off_bh * $(n) * $(d)
    v_offset = off_bh * $(n) * $(e)
    o_offset = off_bh * $(n) * $(e)
    block_offset = off_block * $(BLOCK) + tl.arange(0, $(BLOCK))
    Q_trans_block_ptr = Q + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    K_block_ptr = K + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    V_trans_block_ptr = V + v_offset + block_offset[None, :] * $(e) + tl.arange(0, $(e))[:, None]
    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    DK_trans_block_ptr = DK + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    DV_block_ptr = DV + v_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    array = tl.toReal(tl.arange(0, $(BLOCK)))
    index = array[:, None] - array[None, :]
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < $(n), other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < $(n), other=0.0).to(tl.float32)
    go = tl.load(DO_block_ptr, mask=block_offset[:, None] < $(n), other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < $(n), other=0.0).to(tl.float32)
    dqk = tl.dot(go, v_trans)
    dqk = tl.where(index >= 0.0, dqk, 0)
    dq_intra = tl.dot(dqk, k)
    dk_intra_trans = tl.dot(q_trans, dqk)
    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0.0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, go)
    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra
    tl.store(DQ_block_ptr, dq.to(tl.float32), mask=block_offset[:, None] < $(n))
    tl.store(DK_trans_block_ptr, dk_trans.to(tl.float32), mask=block_offset[None, :] < $(n))
    tl.store(DV_block_ptr, dv.to(tl.float32), mask=block_offset[:, None] < $(n))
  }

def inter (p : Parameters) : ComputeKernel :=
  let b := p.B
  let h := p.H
  let n := p.N
  let d := p.D
  let e := p.E
  let BLOCK := p.BT
  let NUM_BLOCK := p.NT
  let BLOCK_MODEL := p.BM
  let CBLOCK := p.CB
  let NUM_CBLOCK := p.NC
  let Q : RegionName := "query"
  let K : RegionName := "key"
  let V : RegionName := "value"
  let DO : RegionName := "incoming"
  let DQ : RegionName := "queryGradient"
  let DK : RegionName := "keyGradient"
  let DV : RegionName := "valueGradient"
  triton {
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32, DO = tl.float32, DQ = tl.float32, DK = tl.float32, DV = tl.float32
    off_bh = tl.program_id(0)
    off_bh % $(h)
    qk_offset = off_bh * $(n) * $(d)
    v_offset = off_bh * $(n) * $(e)
    o_offset = off_bh * $(n) * $(e)
    DQ_block_ptr = DQ + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    K_block_ptr = K + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    V_trans_block_ptr = V + v_offset + tl.arange(0, $(CBLOCK))[None, :] * $(e) + tl.arange(0, $(e))[:, None]
    DO_block_ptr = DO + o_offset + tl.arange(0, $(CBLOCK))[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    off_block1 = tl.arange(0, $(CBLOCK))
    off_block2 = tl.arange(0, $(CBLOCK))
    kv_trans = tl.zeros([$(e), $(d)], dtype=tl.float32)
    for i in range($(NUM_BLOCK)) {
      for j in range($(NUM_CBLOCK)) {
        if i > 0 {
          go = tl.load(DO_block_ptr, mask=off_block1[:, None] < $(n), other=0.0).to(tl.float32)
          dq_inter = tl.dot(go, kv_trans)
          dq = dq_inter + tl.load(DQ_block_ptr, mask=off_block1[:, None] < $(n), other=0.0)
          tl.store(DQ_block_ptr, dq.to(tl.float32), mask=off_block1[:, None] < $(n))
        }
        DQ_block_ptr += $(CBLOCK) * $(d)
        DO_block_ptr += $(CBLOCK) * $(e)
        off_block1 += $(CBLOCK)
      }
      kv_trans_current = tl.zeros([$(e), $(d)], dtype=tl.float32)
      for j in range($(NUM_CBLOCK)) {
        v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] < $(n), other=0.0).to(tl.float32)
        k = tl.load(K_block_ptr, mask=off_block2[:, None] < $(n), other=0.0).to(tl.float32)
        kv_trans_current += tl.dot(v_trans, k)
        K_block_ptr += $(CBLOCK) * $(d)
        V_trans_block_ptr += $(CBLOCK) * $(e)
        off_block2 += $(CBLOCK)
      }
      kv_trans += kv_trans_current
    }
    m = $(NUM_BLOCK) * $(BLOCK)
    off_block1 = m + tl.arange(0, $(CBLOCK))
    off_block2 = m + tl.arange(0, $(CBLOCK))
    Q_trans_block_ptr = Q + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    K_block_ptr = K + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    V_trans_block_ptr = V + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[None, :] * $(e) + tl.arange(0, $(e))[:, None]
    DK_trans_block_ptr = DK + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    DV_block_ptr = DV + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    DO_block_ptr = DO + o_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    dkv = tl.zeros([$(d), $(e)], dtype=tl.float32)
    for i in range($(NUM_BLOCK) - 1, -1, -1) {
      for j in range($(NUM_CBLOCK) - 1, -1, -1) {
        K_block_ptr -= $(CBLOCK) * $(d)
        V_trans_block_ptr -= $(CBLOCK) * $(e)
        DK_trans_block_ptr -= $(CBLOCK) * $(d)
        DV_block_ptr -= $(CBLOCK) * $(e)
        off_block1 -= $(CBLOCK)
        if i < $((Int.ofNat NUM_BLOCK - 1 : Int)) {
          k = tl.load(K_block_ptr, mask=off_block1[:, None] < $(n), other=0.0).to(tl.float32)
          v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None, :] < $(n), other=0.0).to(tl.float32)
          dk_inter_trans = tl.dot(dkv, v_trans)
          dv_inter = tl.dot(k, dkv)
          dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr, mask=off_block1[None, :] < $(n), other=0.0)
          dv = dv_inter + tl.load(DV_block_ptr, mask=off_block1[:, None] < $(n), other=0.0)
          tl.store(DK_trans_block_ptr, dk_trans.to(tl.float32), mask=off_block1[None, :] < $(n))
          tl.store(DV_block_ptr, dv.to(tl.float32), mask=off_block1[:, None] < $(n))
        }
      }
      dkv_current = tl.zeros([$(d), $(e)], dtype=tl.float32)
      for j in range($(NUM_CBLOCK) - 1, -1, -1) {
        DO_block_ptr -= $(CBLOCK) * $(e)
        Q_trans_block_ptr -= $(CBLOCK) * $(d)
        off_block2 -= $(CBLOCK)
        go = tl.load(DO_block_ptr, mask=off_block2[:, None] < $(n), other=0.0).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] < $(n), other=0.0).to(tl.float32)
        dkv_current += tl.dot(q_trans, go)
      }
      dkv += dkv_current
    }
  }

/-- Reuse the common value/output base offset in each complete source. -/
def forwardCandidate (p : Parameters) : ComputeKernel :=
  let b := p.B
  let h := p.H
  let n := p.N
  let d := p.D
  let e := p.E
  let BLOCK := p.BT
  let NUM_BLOCK := p.NT
  let BLOCK_MODEL := p.BM
  let CBLOCK := p.CB
  let NUM_CBLOCK := p.NC
  let Q : RegionName := "query"
  let K : RegionName := "key"
  let V : RegionName := "value"
  let Out : RegionName := "output"
  triton {
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32
    off_bh = tl.program_id(0)
    off_bh % $(h)
    off_e = tl.program_id(1)
    qk_offset = off_bh * $(n) * $(d)
    v_offset = off_bh * $(n) * $(e)
    o_offset = v_offset
    e_offset = off_e * $(BLOCK_MODEL)
    Q_block_ptr = Q + qk_offset + tl.arange(0, $(d))[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, $(d))[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
    off_block = tl.arange(0, $(BLOCK))
    index = off_block[:, None].to(tl.int64) - off_block[None, :].to(tl.int64)
    kv = tl.zeros([$(d), $(BLOCK_MODEL)], dtype=tl.float32)
    for i in range($(NUM_BLOCK)) {
      q = tl.load(Q_block_ptr + off_block[:, None] * $(d), mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
      k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * $(d), mask=off_block[None, :] < $(n), other=0.0).to(tl.float32)
      v = tl.load(V_block_ptr + off_block[:, None] * $(e), mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
      qk = tl.dot(q, k_trans)
      qk = tl.where(index >= $((0 : Int)), qk, 0)
      o_intra = tl.dot(qk, v)
      o_inter = tl.dot(q, kv)
      o = o_intra + o_inter
      tl.store(O_block_ptr + off_block[:, None] * $(e), o.to(tl.float32), mask=off_block[:, None] < $(n))
      kv += tl.dot(k_trans, v)
      off_block += $(BLOCK)
    }
  }


def intraCandidate (p : Parameters) : ComputeKernel :=
  let b := p.B
  let h := p.H
  let n := p.N
  let d := p.D
  let e := p.E
  let BLOCK := p.BT
  let NUM_BLOCK := p.NT
  let BLOCK_MODEL := p.BM
  let CBLOCK := p.CB
  let NUM_CBLOCK := p.NC
  let Q : RegionName := "query"
  let K : RegionName := "key"
  let V : RegionName := "value"
  let DO : RegionName := "incoming"
  let DQ : RegionName := "queryGradient"
  let DK : RegionName := "keyGradient"
  let DV : RegionName := "valueGradient"
  triton {
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32, DO = tl.float32, DQ = tl.float32, DK = tl.float32, DV = tl.float32
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_bh % $(h)
    qk_offset = off_bh * $(n) * $(d)
    v_offset = off_bh * $(n) * $(e)
    o_offset = v_offset
    block_offset = off_block * $(BLOCK) + tl.arange(0, $(BLOCK))
    Q_trans_block_ptr = Q + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    K_block_ptr = K + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    V_trans_block_ptr = V + v_offset + block_offset[None, :] * $(e) + tl.arange(0, $(e))[:, None]
    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    DK_trans_block_ptr = DK + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    DV_block_ptr = DV + v_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    array = tl.toReal(tl.arange(0, $(BLOCK)))
    index = array[:, None] - array[None, :]
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < $(n), other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < $(n), other=0.0).to(tl.float32)
    go = tl.load(DO_block_ptr, mask=block_offset[:, None] < $(n), other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < $(n), other=0.0).to(tl.float32)
    dqk = tl.dot(go, v_trans)
    dqk = tl.where(index >= 0.0, dqk, 0)
    dq_intra = tl.dot(dqk, k)
    dk_intra_trans = tl.dot(q_trans, dqk)
    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0.0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, go)
    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra
    tl.store(DQ_block_ptr, dq.to(tl.float32), mask=block_offset[:, None] < $(n))
    tl.store(DK_trans_block_ptr, dk_trans.to(tl.float32), mask=block_offset[None, :] < $(n))
    tl.store(DV_block_ptr, dv.to(tl.float32), mask=block_offset[:, None] < $(n))
  }


def interCandidate (p : Parameters) : ComputeKernel :=
  let b := p.B
  let h := p.H
  let n := p.N
  let d := p.D
  let e := p.E
  let BLOCK := p.BT
  let NUM_BLOCK := p.NT
  let BLOCK_MODEL := p.BM
  let CBLOCK := p.CB
  let NUM_CBLOCK := p.NC
  let Q : RegionName := "query"
  let K : RegionName := "key"
  let V : RegionName := "value"
  let DO : RegionName := "incoming"
  let DQ : RegionName := "queryGradient"
  let DK : RegionName := "keyGradient"
  let DV : RegionName := "valueGradient"
  triton {
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32, DO = tl.float32, DQ = tl.float32, DK = tl.float32, DV = tl.float32
    off_bh = tl.program_id(0)
    off_bh % $(h)
    qk_offset = off_bh * $(n) * $(d)
    v_offset = off_bh * $(n) * $(e)
    o_offset = v_offset
    DQ_block_ptr = DQ + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    K_block_ptr = K + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    V_trans_block_ptr = V + v_offset + tl.arange(0, $(CBLOCK))[None, :] * $(e) + tl.arange(0, $(e))[:, None]
    DO_block_ptr = DO + o_offset + tl.arange(0, $(CBLOCK))[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    off_block1 = tl.arange(0, $(CBLOCK))
    off_block2 = tl.arange(0, $(CBLOCK))
    kv_trans = tl.zeros([$(e), $(d)], dtype=tl.float32)
    for i in range($(NUM_BLOCK)) {
      for j in range($(NUM_CBLOCK)) {
        if i > 0 {
          go = tl.load(DO_block_ptr, mask=off_block1[:, None] < $(n), other=0.0).to(tl.float32)
          dq_inter = tl.dot(go, kv_trans)
          dq = dq_inter + tl.load(DQ_block_ptr, mask=off_block1[:, None] < $(n), other=0.0)
          tl.store(DQ_block_ptr, dq.to(tl.float32), mask=off_block1[:, None] < $(n))
        }
        DQ_block_ptr += $(CBLOCK) * $(d)
        DO_block_ptr += $(CBLOCK) * $(e)
        off_block1 += $(CBLOCK)
      }
      kv_trans_current = tl.zeros([$(e), $(d)], dtype=tl.float32)
      for j in range($(NUM_CBLOCK)) {
        v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] < $(n), other=0.0).to(tl.float32)
        k = tl.load(K_block_ptr, mask=off_block2[:, None] < $(n), other=0.0).to(tl.float32)
        kv_trans_current += tl.dot(v_trans, k)
        K_block_ptr += $(CBLOCK) * $(d)
        V_trans_block_ptr += $(CBLOCK) * $(e)
        off_block2 += $(CBLOCK)
      }
      kv_trans += kv_trans_current
    }
    m = $(NUM_BLOCK) * $(BLOCK)
    off_block1 = m + tl.arange(0, $(CBLOCK))
    off_block2 = m + tl.arange(0, $(CBLOCK))
    Q_trans_block_ptr = Q + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    K_block_ptr = K + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[:, None] * $(d) + tl.arange(0, $(d))[None, :]
    V_trans_block_ptr = V + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[None, :] * $(e) + tl.arange(0, $(e))[:, None]
    DK_trans_block_ptr = DK + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) + tl.arange(0, $(d))[:, None]
    DV_block_ptr = DV + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    DO_block_ptr = DO + o_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) + tl.arange(0, $(e))[None, :]
    dkv = tl.zeros([$(d), $(e)], dtype=tl.float32)
    for i in range($(NUM_BLOCK) - 1, -1, -1) {
      for j in range($(NUM_CBLOCK) - 1, -1, -1) {
        K_block_ptr -= $(CBLOCK) * $(d)
        V_trans_block_ptr -= $(CBLOCK) * $(e)
        DK_trans_block_ptr -= $(CBLOCK) * $(d)
        DV_block_ptr -= $(CBLOCK) * $(e)
        off_block1 -= $(CBLOCK)
        if i < $((Int.ofNat NUM_BLOCK - 1 : Int)) {
          k = tl.load(K_block_ptr, mask=off_block1[:, None] < $(n), other=0.0).to(tl.float32)
          v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None, :] < $(n), other=0.0).to(tl.float32)
          dk_inter_trans = tl.dot(dkv, v_trans)
          dv_inter = tl.dot(k, dkv)
          dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr, mask=off_block1[None, :] < $(n), other=0.0)
          dv = dv_inter + tl.load(DV_block_ptr, mask=off_block1[:, None] < $(n), other=0.0)
          tl.store(DK_trans_block_ptr, dk_trans.to(tl.float32), mask=off_block1[None, :] < $(n))
          tl.store(DV_block_ptr, dv.to(tl.float32), mask=off_block1[:, None] < $(n))
        }
      }
      dkv_current = tl.zeros([$(d), $(e)], dtype=tl.float32)
      for j in range($(NUM_CBLOCK) - 1, -1, -1) {
        DO_block_ptr -= $(CBLOCK) * $(e)
        Q_trans_block_ptr -= $(CBLOCK) * $(d)
        off_block2 -= $(CBLOCK)
        go = tl.load(DO_block_ptr, mask=off_block2[:, None] < $(n), other=0.0).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] < $(n), other=0.0).to(tl.float32)
        dkv_current += tl.dot(q_trans, go)
      }
      dkv += dkv_current
    }
  }

end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
