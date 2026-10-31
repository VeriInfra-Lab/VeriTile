import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete typed repaired source. The selected public ABI instantiates FP32
storage throughout. Seven checked block loads explicitly zero-pad. H/DH select
the history chunk in the base and mask against logical K, so a final key tile
cannot read the following chunk. The mathematical variant flag is retained only
for internal inherited lemmas; the public kernel selects the unchanged original
arithmetic order. The source returns per-key-tile gate contributions and leaves
reverse cumulative sum to its separate downstream kernel. -/
open VeriTile.Triton
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSource

set_option maxRecDepth 40000

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

/-- Complete ordinary source quotations for the existing storage interface.
The public selected ABI is FP32. No whole-source macro or dropped statement. -/
def specialize (qk value h dh : Storage) (swapped : Bool) :=
  let dtype := qk.dtype
  let out_dtype := value.dtype
  match h, dh, swapped with
  | .fp32, .fp32, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.float32,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp32, .fp32, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.float32,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp32, .fp16, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.float16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp32, .fp16, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.float16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp32, .bf16, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.bfloat16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp32, .bf16, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.bfloat16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp16, .fp32, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float16, dh = tl.float32,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp16, .fp32, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float16, dh = tl.float32,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp16, .fp16, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float16, dh = tl.float16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp16, .fp16, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float16, dh = tl.float16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp16, .bf16, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float16, dh = tl.bfloat16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .fp16, .bf16, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float16, dh = tl.bfloat16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .bf16, .fp32, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.bfloat16, dh = tl.float32,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .bf16, .fp32, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.bfloat16, dh = tl.float32,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .bf16, .fp16, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.bfloat16, dh = tl.float16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .bf16, .fp16, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.bfloat16, dh = tl.float16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .bf16, .bf16, Bool.false =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.bfloat16, dh = tl.bfloat16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq += tl.dot(b_ds, b_k)
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

  | .bf16, .bf16, Bool.true =>
    fun (q k v h g do_ dh dq dk dg : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV NT : Nat) =>
    triton {
      tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
      tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
        v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.bfloat16, dh = tl.bfloat16,
        g = tl.float32, dg = tl.float32

      i_k = tl.program_id(0)
      i_t = tl.program_id(1)
      i_bh = tl.program_id(2)
      n_bh = tl.num_programs(2)
      o_i = tl.arange(0, $(BT))
      p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
      last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
      b_g_last = tl.load(g + i_bh * $(T) + last_idx)
      b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
      b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
      b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
      b_dg = tl.zeros([$(BT)], dtype=tl.float32)
      for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
          shape=($(V), $(K)), strides=($(1), $(s_h_t)),
          offsets=(i_v * $(BV), i_k * $(BK)),
          block_shape=($(BV), $(BK)), order=(0, 1))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
        b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
      }
      p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last = b_dg_last * tl.exp(b_g_last)
      b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
      b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
      b_dg_last += tl.sum(b_dk * b_k)
      b_ds = tl.where(o_i[:, None] >= o_i[None, :],
        b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
      b_ds = (b_ds).to(DTYPE)
      b_dq = tl.dot(b_ds, b_k) + b_dq
      b_dk += tl.dot(tl.trans(b_ds), b_q)
      b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
      b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
      p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
        shape=($(T), $(K)), strides=($(s_k_t), $(1)),
        offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
      p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
        strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
      tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
      tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
    }

def original (qk value h dh : Storage) := specialize qk value h dh Bool.false

/-- Selected FP32 candidate exposes the same absolute chunk end already
computed for last_idx, permitting reuse in the final gate mask. All floating
operations and their provenance remain unchanged. -/
def tailBoundCSE :=
  let dtype := FloatDType.fp32
  let out_dtype := FloatDType.fp32
  fun (q k v h g do_ dh dq dk dg : RegionName)
      (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
      (scale : ℝ) (T K V BT BK BV NT : Nat) =>
  triton {
    tl.ports inputs=[q, k, v, h, g, do_, dh], outputs=[dq, dk, dg]
    tl.region q = DTYPE, k = DTYPE, dq = DTYPE, dk = DTYPE,
      v = OUT_DTYPE, do_ = OUT_DTYPE, h = tl.float32, dh = tl.float32,
      g = tl.float32, dg = tl.float32

    i_k = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, $(BT))
    p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    b_g = tl.load(p_g, boundary_check=([0] : List Nat), padding_option="zero")
    last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
    b_g_last = tl.load(g + i_bh * $(T) + last_idx)
    b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
    b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
    b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
    b_dg_last = tl.zeros([$(1)], dtype=tl.float32)
    b_dg = tl.zeros([$(BT)], dtype=tl.float32)
    for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
      p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
        shape=($(T), $(V)), strides=($(s_v_t), $(1)),
        offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
      p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
        shape=($(V), $(K)), strides=($(1), $(s_h_t)),
        offsets=(i_v * $(BV), i_k * $(BK)),
        block_shape=($(BV), $(BK)), order=(0, 1))
      p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
        shape=($(T), $(V)), strides=($(s_v_t), $(1)),
        offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
      p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K * s_h_t),
        shape=($(V), $(K)), strides=($(1), $(s_h_t)),
        offsets=(i_v * $(BV), i_k * $(BK)),
        block_shape=($(BV), $(BK)), order=(0, 1))
      b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat), padding_option="zero")
      b_dg_last += tl.sum(b_h * b_dh)
      b_ds += tl.dot(b_do, tl.trans(b_v))
      b_dq += tl.dot(b_do, (b_h).to(OUT_DTYPE))
      b_dk += tl.dot(b_v, (b_dh).to(OUT_DTYPE))
    }
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat), padding_option="zero")
    b_dg_last = b_dg_last * tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :],
      b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
    b_ds = (b_ds).to(DTYPE)
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    b_dg = tl.where(o_i < (min(i_t * $(BT) + $(BT), $(T)) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    tl.store(p_dq, (b_dq).to(DTYPE), boundary_check=([0, 1] : List Nat))
    tl.store(p_dk, (b_dk).to(DTYPE), boundary_check=([0, 1] : List Nat))
    tl.store(p_dg, (b_dg).to(tl.float32), boundary_check=([0] : List Nat))
  }

end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSource
