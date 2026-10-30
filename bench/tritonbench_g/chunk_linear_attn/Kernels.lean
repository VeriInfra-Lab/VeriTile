import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile.Triton
structure Parameters where
  (B H T K V BT BK BV NT : Nat)

/- Full repaired selected FP32 sources. BT/BK/BV are supplied by exact host
metadata; reverse uses ascending j and NT-1-j, preserving descending chunks.
All masks, casts and source operation order are retained. -/
def forwardH (p : Parameters) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let k : RegionName := "key"
  let v : RegionName := "value"
  let h : RegionName := "history"
  let h0 : RegionName := "initial"
  let ht : RegionName := "final"
  triton {
        tl.region k = tl.float32, v = tl.float32, h = tl.float32, h0 = tl.float32, ht = tl.float32
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=h0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat), padding_option="zero")).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h += tl.dot(b_k, b_v, allow_tf32=False)
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
        }
      }

def forwardO (p : Parameters) (scale : ℝ) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let k : RegionName := "key"
  let v : RegionName := "value"
  let h : RegionName := "history"
  let o : RegionName := "output"
  triton {
        tl.region q = tl.float32, k = tl.float32, h = tl.float32, v = tl.float32, o = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        o_i = tl.arange(0, $(BT))
        m_s = o_i[:, None] >= o_i[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_o += tl.dot(b_q, b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s = tl.where(m_s, b_s, 0)
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_o = (b_o + tl.dot((b_s).to(tl.float32), b_v, allow_tf32=False)) * $((scale : ℝ))
        tl.store(p_o, (b_o).to(tl.float32), boundary_check=([0, 1] : List Nat))
      }

def backwardH (p : Parameters) (scale : ℝ) (USE_FINAL_GRADIENT STORE_INITIAL_GRADIENT : Bool) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let do_ : RegionName := "incoming"
  let dh : RegionName := "historyGradient"
  let dht : RegionName := "terminalGradient"
  let dh0 : RegionName := "initialGradient"
  triton {
        tl.region q = tl.float32, do_ = tl.float32, dh = tl.float32, dht = tl.float32, dh0 = tl.float32
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_FINAL_GRADIENT {
          p_dht = tl.make_block_ptr(base=dht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_dh = (tl.load(p_dht, boundary_check=([0, 1] : List Nat), padding_option="zero")).to(tl.float32)
        }
        for j in range($(0), $(NT), $(1)) {
          i_t = $(NT) - $(1) - j
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=(dh + (i_bh * $(s_h_h))) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_dh, (b_dh).to(tl.float32), boundary_check=([0, 1] : List Nat))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_q = ((b_q * $((scale : ℝ)))).to(tl.float32)
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_dh += tl.dot(b_q, (b_do).to(tl.float32), allow_tf32=False)
        }
        if STORE_INITIAL_GRADIENT {
          p_dh0 = tl.make_block_ptr(base=dh0 + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_dh0, (b_dh).to(tl.float32), boundary_check=([0, 1] : List Nat))
        }
      }

def backwardQKV (p : Parameters) (scale : ℝ) : ComputeKernel :=
  let T := p.T
  let K := p.K
  let V := p.V
  let BT := p.BT
  let BK := p.BK
  let BV := p.BV
  let NT := p.NT
  let s_qk_h : Nat := p.T*p.K
  let s_qk_t : Nat := p.K
  let s_qk_d : Nat := 1
  let s_vo_h : Nat := p.T*p.V
  let s_vo_t : Nat := p.V
  let s_vo_d : Nat := 1
  let s_h_h : Nat := p.NT*p.K*p.V
  let s_h_t : Nat := p.V
  let q : RegionName := "query"
  let k : RegionName := "key"
  let v : RegionName := "value"
  let h : RegionName := "history"
  let do_ : RegionName := "incoming"
  let dh : RegionName := "historyGradient"
  let dq : RegionName := "queryGradient"
  let dk : RegionName := "keyGradient"
  let dv : RegionName := "valuePartials"
  triton {
        tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, do_ = tl.float32, dh = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        n_bh = tl.num_programs(2)
        o_i = tl.arange(0, $(BT))
        p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
        p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_s = (tl.dot(b_k, b_q, allow_tf32=False) * $((scale : ℝ)))
        b_s = tl.where(o_i[:, None] <= o_i[None, :], b_s, 0)
        b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_v in range(tl.cdiv($(V), $(BV))) {
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)), shape=($(V), $(NT * K)), strides=(1, $(s_h_t)), offsets=((i_v * $(BV)), ((i_t * $(K)) + (i_k * $(BK)))), block_shape=($(BV), $(BK)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)), shape=($(NT * K), $(V)), strides=($(s_h_t), 1), offsets=(((i_t * $(K)) + (i_k * $(BK))), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          p_dv = tl.make_block_ptr(base=dv + (((i_k * n_bh) + i_bh) * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
          b_dq += (tl.dot(b_do, b_h, allow_tf32=False) * $((scale : ℝ)))
          b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
          b_dv = (tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot((b_s).to(tl.float32), b_do, allow_tf32=False))
          tl.store(p_dv, (b_dv).to(tl.float32), boundary_check=([0, 1] : List Nat))
        }
        b_ds = (tl.where(o_i[:, None] >= o_i[None, :], (b_ds * $((scale : ℝ))), 0)).to(tl.float32)
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
        p_dq = tl.make_block_ptr(base=dq + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        p_dk = tl.make_block_ptr(base=dk + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        tl.store(p_dq, (b_dq).to(tl.float32), boundary_check=([0, 1] : List Nat))
        tl.store(p_dk, (b_dk).to(tl.float32), boundary_check=([0, 1] : List Nat))
      }

end VeriTile.Bench.Repairs.ChunkLinearAttn
