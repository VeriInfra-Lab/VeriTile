import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Typed source for the separate wide-pointer repair. The actual Python
uses signed-int64 program/chunk indices; this source uses exact natural indices
whose refinement is established in Proofs/Address. Native block-pointer offsets
are all zero and parents are exactly one local tile. The flattened program ID
is decoded before the unchanged floating body. All source dtype combinations,
casts, dot options, branches and separate mul/add operations are retained.
The indexFirst/sumLeft forms remain the original optional local FP rewrites. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSource
open VeriTile.Triton

def cumOriginal :=
  fun (dtype out_dtype : FloatDType) (s o : RegionName)
      (s_s_h s_s_t s_s_d T S BT BS BH PROGRAM_BASE : Nat) =>
    triton {
      tl.ports inputs=[s], outputs=[o]
      tl.region s = DTYPE, o = OUT_DTYPE
      linear = tl.program_id(0) + $(PROGRAM_BASE)
      if linear < $(BH * (T / BT) * (S / BS)) {
        i_s = linear % $(S / BS)
        i_t = (linear / $(S / BS)) % $(T / BT)
        i_bh = linear / $((S / BS) * (T / BT))
        o_i = tl.arange(0, $(BT))
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
        p_s = tl.make_block_ptr(base=s + i_bh * $(s_s_h) + i_t * $(BT) * $(s_s_t) + i_s * $(BS) * $(s_s_d),
          shape=($(BT), $(BS)), strides=($(s_s_t), $(s_s_d)),
          offsets=($(0), $(0)), block_shape=($(BT), $(BS)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_s_h) + i_t * $(BT) * $(s_s_t) + i_s * $(BS) * $(s_s_d),
          shape=($(BT), $(BS)), strides=($(s_s_t), $(s_s_d)),
          offsets=($(0), $(0)), block_shape=($(BT), $(BS)), order=(1, 0))
        b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
        b_o = tl.dot(m_s, b_s, allow_tf32=false)
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }
    }

def cumIndexFirst :=
  fun (dtype out_dtype : FloatDType) (s o : RegionName)
      (s_s_h s_s_t s_s_d T S BT BS BH PROGRAM_BASE : Nat) =>
    triton {
      tl.ports inputs=[s], outputs=[o]
      tl.region s = DTYPE, o = OUT_DTYPE
      linear = tl.program_id(0) + $(PROGRAM_BASE)
      if linear < $(BH * (T / BT) * (S / BS)) {
        i_s = linear % $(S / BS)
        i_t = (linear / $(S / BS)) % $(T / BT)
        i_bh = linear / $((S / BS) * (T / BT))
        o_i = tl.arange(0, $(BT))
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
        p_s = tl.make_block_ptr(base=s + $(s_s_h) * i_bh + i_t * $(BT) * $(s_s_t) + i_s * $(BS) * $(s_s_d),
          shape=($(BT), $(BS)), strides=($(s_s_t), $(s_s_d)),
          offsets=($(0), $(0)), block_shape=($(BT), $(BS)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + $(s_s_h) * i_bh + i_t * $(BT) * $(s_s_t) + i_s * $(BS) * $(s_s_d),
          shape=($(BT), $(BS)), strides=($(s_s_t), $(s_s_d)),
          offsets=($(0), $(0)), block_shape=($(BT), $(BS)), order=(1, 0))
        b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
        b_o = tl.dot(m_s, b_s, allow_tf32=false)
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }
    }

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

/-! Each existing storage specialization below is a complete ordinary quotation.
The two branches differ only at the final carry/dot addition. -/
set_option maxHeartbeats 3000000 in
def hSource (dotStorage gateStorage historyStorage : Storage) (swapped : Bool) :=
  match dotStorage, gateStorage, historyStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float32, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float32, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .fp32, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float32, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float32, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .fp32, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float32, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float32, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .fp16, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .fp16, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .fp16, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.float16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .bf16, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.bfloat16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.bfloat16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .bf16, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.bfloat16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.bfloat16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp32, .bf16, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.bfloat16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float32, v = tl.float32, g = tl.bfloat16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float32)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float32)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .fp32, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float32, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float32, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .fp32, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float32, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float32, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .fp32, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float32, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float32, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .fp16, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .fp16, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .fp16, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.float16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .bf16, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.bfloat16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.bfloat16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .bf16, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.bfloat16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.bfloat16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .fp16, .bf16, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.bfloat16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.float16, v = tl.float16, g = tl.bfloat16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.float16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.float16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .fp32, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float32, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float32, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .fp32, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float32, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float32, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .fp32, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float32, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float32, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .fp16, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .fp16, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .fp16, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.float16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .bf16, .fp32 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.bfloat16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.bfloat16, h = tl.float32,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float32), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .bf16, .fp16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.bfloat16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.bfloat16, h = tl.float16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.float16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
  | .bf16, .bf16, .bf16 =>
      if swapped then fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.bfloat16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h = tl.dot(b_k, b_v, allow_tf32=false) + b_h
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }
      else fun (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
            (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
              T K V BT BK BV NT PROGRAM_BASE : Nat) (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
          triton {
            tl.ports inputs=[k, v, g, h0], outputs=[h, ht]
            tl.region k = tl.bfloat16, v = tl.bfloat16, g = tl.bfloat16, h = tl.bfloat16,
              h0 = DTYPE, ht = OUT_DTYPE
            linear = tl.program_id(0) + $(PROGRAM_BASE)
            i_v = linear % $(V / BV)
            i_k = (linear / $(V / BV)) % $(K / BK)
            i_bh = linear / $((V / BV) * (K / BK))
            b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
            if USE_INITIAL_STATE {
              p_h = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              b_h += tl.load(p_h, boundary_check=([0, 1] : List Nat)).to(tl.float32)
            }
            for i_t in range($(0), $(NT), $(1)) {
              p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
              p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
              p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V) + i_k * $(BK) * $(s_h_t) + i_v * $(BV) * $(s_h_d),
                shape=($(BK), $(BV)), strides=($(s_h_t), $(s_h_d)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(tl.bfloat16), boundary_check=([0, 1] : List Nat))
              b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
              b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
              if GATEK {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + i_k * $(BK) * $(s_k_d) + i_t * $(BT) * $(s_k_t),
                  shape=($(BK), $(BT)), strides=($(s_k_d), $(s_k_t)),
                  offsets=($(0), $(0)), block_shape=($(BK), $(BT)), order=(0, 1))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_k_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_k_t) + i_k * $(BK) * $(s_k_d),
                  shape=($(BK)), strides=($(s_k_d)),
                  offsets=($(0)),
                  block_shape=($(BK)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[:, None]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(tl.bfloat16)
              } else {
                p_g = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + i_t * $(BT) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BT), $(BV)), strides=($(s_v_t), $(s_v_d)),
                  offsets=($(0), $(0)), block_shape=($(BT), $(BV)), order=(1, 0))
                p_gn = tl.make_block_ptr(base=g + i_bh * $(s_v_h) + (i_t * $(BT) + $(BT) - $(1)) * $(s_v_t) + i_v * $(BV) * $(s_v_d),
                  shape=($(BV)), strides=($(s_v_d)),
                  offsets=($(0)),
                  block_shape=($(BV)), order=(0))
                b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
                b_h = b_h * tl.exp(b_gn)[None, :]
                b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
                b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(tl.bfloat16)
              }
              b_h += tl.dot(b_k, b_v, allow_tf32=false)
            }
            if STORE_FINAL_STATE {
              p_h = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V) + i_k * $(BK) * $(V) + i_v * $(BV),
                shape=($(BK), $(BV)), strides=($(V), $(1)),
                offsets=($(0), $(0)), block_shape=($(BK), $(BV)), order=(1, 0))
              tl.store(p_h, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
            }
          }

def hOriginal (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.false

def hSumLeft (dotStorage gateStorage historyStorage : Storage) :=
  hSource dotStorage gateStorage historyStorage Bool.true

end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSource

namespace VeriTile.Bench.Repairs.ChunkGatedAttention
abbrev Original := VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSource.Storage
def originalPythonSHA256 : String :=
  "8f67a536f4c7850f6025debc90d692e3d8e007159e92df38ea144c20eac77b1f"
abbrev cumulativeKernel := NativeSource.cumOriginal
abbrev recurrentKernel := NativeSource.hOriginal
def defaultStorage : Original := .fp32
end VeriTile.Bench.Repairs.ChunkGatedAttention
