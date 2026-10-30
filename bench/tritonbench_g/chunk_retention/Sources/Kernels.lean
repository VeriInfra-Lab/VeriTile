import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Selected FP32 typed source transcription of the separate repaired four
kernels. Signed lane casts state the implicit source integer lane type.
Reverse iteration is represented by NT-1-j. The complete native source binding
is exposed separately; this file is not a generic compiler-correctness proof. -/
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Kernels
open Lean VeriTile.Triton
set_option maxHeartbeats 8000000

def forwardH (k v h initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :=
      triton {
        tl.ports inputs=[k,v,initial_state], outputs=[h,final_state]
        tl.region k = tl.float32, v = tl.float32, h = tl.float32, initial_state = tl.float32, final_state = tl.float32
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = (tl.program_id(2)).to(tl.int64)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=initial_state + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat), padding_option="zero")).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(tl.float32), allow_tf32=False)
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=final_state + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
        }
      }

def forwardO (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat) :=
      triton {
        tl.ports inputs=[q,k,v,h], outputs=[o]
        tl.region q = tl.float32, k = tl.float32, h = tl.float32, v = tl.float32, o = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = (tl.program_id(2)).to(tl.int64)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_i = tl.math.exp2((o_i + $(1)) * b_b)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_o += tl.dot((b_q * d_i[:, None]).to(tl.float32), b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s *= d_s
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_o = (b_o + tl.dot((b_s).to(tl.float32), b_v, allow_tf32=False)) * $((scale : ℝ))
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        tl.store(p_o, (b_o).to(tl.float32), boundary_check=([0, 1] : List Nat))
      }

/-- Existing scale-first candidate: the final FP32 multiplication commutes its operands. -/
def forwardOScaleFirst (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat) :=
      triton {
        tl.ports inputs=[q,k,v,h], outputs=[o]
        tl.region q = tl.float32, k = tl.float32, h = tl.float32, v = tl.float32, o = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = (tl.program_id(2)).to(tl.int64)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_i = tl.math.exp2((o_i + $(1)) * b_b)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_o += tl.dot((b_q * d_i[:, None]).to(tl.float32), b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s *= d_s
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_o = $((scale : ℝ)) * (b_o + tl.dot((b_s).to(tl.float32), b_v, allow_tf32=False))
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        tl.store(p_o, (b_o).to(tl.float32), boundary_check=([0, 1] : List Nat))
      }

def backwardQKV  (q k v h do_ dh dq dk dv : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat) :=
      triton {
        tl.ports inputs=[q,k,v,h,do_,dh], outputs=[dq,dk,dv]
        tl.region q = tl.float32, k = tl.float32, v = tl.float32, h = tl.float32, do_ = tl.float32, dh = tl.float32, dq = tl.float32, dk = tl.float32, dv = tl.float32
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = (tl.program_id(2)).to(tl.int64)
        i_h = i_bh % $(H)
        n_bh = (tl.num_programs(2)).to(tl.int64)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_q = tl.math.exp2((o_i + $(1)) * b_b)
        d_k = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
          d_k = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
        }
        d_q = (d_q * $((scale : ℝ))).to(tl.float32)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * $((scale : ℝ))
        p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
        p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_s = (tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s))
        b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_v in range(tl.cdiv($(V), $(BV))) {
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(V), $(K)), strides=(1, $(s_h_t)), offsets=((i_v * $(BV)), (i_k * $(BK))), block_shape=($(BV), $(BK)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          p_dv = tl.make_block_ptr(base=dv + (((i_k * n_bh) + i_bh) * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
          b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
          b_dq += tl.dot(b_do, b_h, allow_tf32=False)
          b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
          b_dv = (tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot((b_s).to(tl.float32), b_do, allow_tf32=False))
          tl.store(p_dv, (b_dv).to(tl.float32), boundary_check=([0, 1] : List Nat))
        }
        b_ds = (b_ds * d_s).to(tl.float32)
        b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
        b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
        p_dq = tl.make_block_ptr(base=dq + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        p_dk = tl.make_block_ptr(base=dk + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        tl.store(p_dq, (b_dq).to(tl.float32), boundary_check=([0, 1] : List Nat))
        tl.store(p_dk, (b_dk).to(tl.float32), boundary_check=([0, 1] : List Nat))
      }

def backwardH (q do_ dh d_ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (H T K V BT BK BV NT : Nat) (USE_FINAL_GRAD : Bool) :=
  triton {
    tl.ports inputs=[q,do_,d_ht], outputs=[dh]
    tl.region q = tl.float32, do_ = tl.float32, dh = tl.float32, d_ht = tl.float32
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = (tl.program_id(2)).to(tl.int64)
    i_h = i_bh % $(H)
    b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
    o_i = (tl.arange(0, $(BT))).to(tl.int32)
    d_b = tl.math.exp2($(BT) * b_b)
    d_i = tl.math.exp2((o_i + $(1)) * b_b)
    b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
    if USE_FINAL_GRAD {
      p_dht = tl.make_block_ptr(base=d_ht + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
      b_dh = (tl.load(p_dht, boundary_check=([0, 1] : List Nat), padding_option="zero")).to(tl.float32)
    }
    for j in range($(0), $(NT), $(1)) {
      i_t = $(NT) - $(1) - j
      p_o = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
      p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
      p_h = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
      tl.store(p_h, (b_dh).to(tl.float32), boundary_check=([0, 1] : List Nat))
      b_o = tl.load(p_o, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
        d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
      } else {
        d_b = tl.math.exp2($(BT) * b_b)
      }
      b_dh = d_b * b_dh + tl.dot(b_q, (b_o * d_i[:, None] * $((scale : ℝ))).to(tl.float32), allow_tf32=False)
    }
  }

end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Kernels
#guard_msgs (drop info) in
#auditModuleAxioms
