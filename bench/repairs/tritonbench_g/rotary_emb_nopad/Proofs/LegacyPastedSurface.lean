import bench.repairs.tritonbench_g.rotary_emb_nopad.Kernels

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Kernels.LegacyPastedSurface
open VeriTile.Triton

/-- Complete transcription of `rotary_embedding_kernel`; FP32 payloads and signed int64 address arithmetic. -/
def uncached (p : UncachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  triton {
    tl.region q = tl.float32, k = tl.float32, cos = tl.float32, sin = tl.float32
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_block_idx = tl.program_id(1).to(tl.int64)
    tokens_range = cur_token_block_idx * $((p.BLOCK_TOKENS : Int)) +
      tl.arange(0, $(p.BLOCK_TOKENS)).to(tl.int64)
    dim_range0 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64)
    dim_range1 = dim_range0 + $(((p.HEAD_DIM / 2) : Int))
    off_cos_sin = tokens_range[:, None] * $((p.cos_token_stride : Int)) +
      dim_range0[None, :] * $((p.cos_stride : Int))
    loaded_cos = tl.load(cos + off_cos_sin,
      mask=(tokens_range[:, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    off_sin = tokens_range[:, None] * $((p.sin_token_stride : Int)) +
      dim_range0[None, :] * $((p.sin_stride : Int))
    loaded_sin = tl.load(sin + off_sin,
      mask=(tokens_range[:, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    off_q0 = tokens_range[:, None, None] * $((p.q_token_stride : Int)) +
      cur_head_idx * $((p.q_head_stride : Int)) +
      dim_range0[None, None, :] * $((p.head_dim_stride : Int))
    off_q1 = tokens_range[:, None, None] * $((p.q_token_stride : Int)) +
      cur_head_idx * $((p.q_head_stride : Int)) +
      dim_range1[None, None, :] * $((p.head_dim_stride : Int))
    loaded_q0 = tl.load(q + off_q0,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    loaded_q1 = tl.load(q + off_q1,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]
    tl.store(q + off_q0, out_q0,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
    tl.store(q + off_q1, out_q1,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
    handle_kv = cur_head_idx % $((p.KV_GROUP_NUM : Int)) == $((0 : Int))
    if handle_kv {
      k_head_idx = cur_head_idx // $((p.KV_GROUP_NUM : Int))
      off_k0 = tokens_range[:, None, None] * $((p.k_token_stride : Int)) +
        k_head_idx * $((p.k_head_stride : Int)) +
        dim_range0[None, None, :] * $((p.head_dim_stride : Int))
      off_k1 = tokens_range[:, None, None] * $((p.k_token_stride : Int)) +
        k_head_idx * $((p.k_head_stride : Int)) +
        dim_range1[None, None, :] * $((p.head_dim_stride : Int))
      loaded_k0 = tl.load(k + off_k0,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      loaded_k1 = tl.load(k + off_k1,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
      out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
      tl.store(k + off_k0, out_k0,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
      tl.store(k + off_k1, out_k1,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
    }
  }

/-- Complete transcription of `fused_rotary_embedding_kernel_v2`; FP32 payloads and signed int64 address arithmetic. -/
def cached (p : CachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  let kv_cache := p.kv_cache
  let BLOCK_TABLES := p.BLOCK_TABLES
  let context_lengths := p.context_lengths
  triton {
    tl.region q = tl.float32, k = tl.float32, cos = tl.float32, sin = tl.float32, kv_cache = tl.float32, BLOCK_TABLES = tl.int64, context_lengths = tl.int64
    block_head_index = tl.program_id(0).to(tl.int64)
    if block_head_index < $((p.Q_HEAD_NUM : Int)) {
      block_token_index = tl.program_id(1).to(tl.int64)
      dim_range0 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64)
      dim_range1 = dim_range0 + $(((p.HEAD_DIM / 2) : Int))
      off_q0 = block_token_index * $((p.q_token_stride : Int)) +
        block_head_index * $((p.q_head_stride : Int)) +
        dim_range0 * $((p.head_dim_stride : Int))
      off_q1 = block_token_index * $((p.q_token_stride : Int)) +
        block_head_index * $((p.q_head_stride : Int)) +
        dim_range1 * $((p.head_dim_stride : Int))
      off_k0 = block_token_index * $((p.k_token_stride : Int)) +
        block_head_index * $((p.k_head_stride : Int)) +
        dim_range0 * $((p.head_dim_stride : Int))
      off_k1 = block_token_index * $((p.k_token_stride : Int)) +
        block_head_index * $((p.k_head_stride : Int)) +
        dim_range1 * $((p.head_dim_stride : Int))
      loaded_q0 = tl.load(q + off_q0, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)), other=0.0)
      loaded_q1 = tl.load(q + off_q1, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)), other=0.0)
      loaded_k0 = tl.load(k + off_k0,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      loaded_k1 = tl.load(k + off_k1,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      off_cos_sin = block_token_index * $((p.cos_token_stride : Int)) + dim_range0 * $((p.cos_stride : Int))
      loaded_cos = tl.load(cos + off_cos_sin,
        mask=(block_token_index < $((p.q_total_tokens : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      off_sin = block_token_index * $((p.sin_token_stride : Int)) + dim_range0 * $((p.sin_stride : Int))
      loaded_sin = tl.load(sin + off_sin,
        mask=(block_token_index < $((p.q_total_tokens : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
      out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
      out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
      out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
      past_kv_seq_len = tl.load(context_lengths + block_token_index) - $((1 : Int))
      last_block_idx = past_kv_seq_len // $((p.block_size : Int))
      block_table_ptr = BLOCK_TABLES + block_token_index * $((p.bts_stride : Int))
      block_ids = tl.load(block_table_ptr +
        last_block_idx * $((p.btb_stride : Int)),
        mask=block_token_index < $((p.q_total_tokens : Int)))
      offsets_in_last_block = past_kv_seq_len % $((p.block_size : Int)) * $((p.cachebs_stride : Int))
      kv_range0 = block_ids * $((p.cacheb_stride : Int)) +
        block_head_index * $((p.cacheh_stride : Int)) +
        offsets_in_last_block +
        dim_range0 * $((p.cached_stride : Int))
      kv_range1 = block_ids * $((p.cacheb_stride : Int)) +
        block_head_index * $((p.cacheh_stride : Int)) +
        offsets_in_last_block +
        dim_range1 * $((p.cached_stride : Int))
      tl.store(kv_cache + kv_range0, out_k0,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))))
      tl.store(kv_cache + kv_range1, out_k1,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))))
      tl.store(q + off_q0, out_q0, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)))
      tl.store(q + off_q1, out_q1, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)))
    }
  }

/-- Complete uncached candidate reusing the computed low-half Q address for
its high-half address. This removes repeated token/head stride arithmetic;
the GQA branch, all masks, numerical operations and stores remain intact. -/
def uncachedShared (p : UncachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  triton {
    tl.region q = tl.float32, k = tl.float32, cos = tl.float32, sin = tl.float32
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_block_idx = tl.program_id(1).to(tl.int64)
    tokens_range = cur_token_block_idx * $((p.BLOCK_TOKENS : Int)) +
      tl.arange(0, $(p.BLOCK_TOKENS)).to(tl.int64)
    dim_range0 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64)
    dim_range1 = dim_range0 + $(((p.HEAD_DIM / 2) : Int))
    off_cos_sin = tokens_range[:, None] * $((p.cos_token_stride : Int)) +
      dim_range0[None, :] * $((p.cos_stride : Int))
    loaded_cos = tl.load(cos + off_cos_sin,
      mask=(tokens_range[:, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    off_sin = tokens_range[:, None] * $((p.sin_token_stride : Int)) +
      dim_range0[None, :] * $((p.sin_stride : Int))
    loaded_sin = tl.load(sin + off_sin,
      mask=(tokens_range[:, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    off_q0 = tokens_range[:, None, None] * $((p.q_token_stride : Int)) +
      cur_head_idx * $((p.q_head_stride : Int)) +
      dim_range0[None, None, :] * $((p.head_dim_stride : Int))
    off_q1 = off_q0 + $(((p.HEAD_DIM / 2) : Int)) * $((p.head_dim_stride : Int))
    loaded_q0 = tl.load(q + off_q0,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    loaded_q1 = tl.load(q + off_q1,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
      other=0.0)
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]
    tl.store(q + off_q0, out_q0,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
    tl.store(q + off_q1, out_q1,
      mask=(cur_head_idx < $((p.Q_HEAD_NUM : Int))) &
      (tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
      (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
    handle_kv = cur_head_idx % $((p.KV_GROUP_NUM : Int)) == $((0 : Int))
    if handle_kv {
      k_head_idx = cur_head_idx // $((p.KV_GROUP_NUM : Int))
      off_k0 = tokens_range[:, None, None] * $((p.k_token_stride : Int)) +
        k_head_idx * $((p.k_head_stride : Int)) +
        dim_range0[None, None, :] * $((p.head_dim_stride : Int))
      off_k1 = tokens_range[:, None, None] * $((p.k_token_stride : Int)) +
        k_head_idx * $((p.k_head_stride : Int)) +
        dim_range1[None, None, :] * $((p.head_dim_stride : Int))
      loaded_k0 = tl.load(k + off_k0,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      loaded_k1 = tl.load(k + off_k1,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
      out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
      tl.store(k + off_k0, out_k0,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
      tl.store(k + off_k1, out_k1,
        mask=(tokens_range[:, None, None] < $((p.q_total_tokens : Int))) &
        (dim_range0[None, None, :] < $(((p.HEAD_DIM / 2) : Int))))
    }
  }

/-- The directly related cached candidate performs the same Q-address CSE.
All signed metadata loads, cache destinations and cache-before-Q stores remain. -/
def cachedShared (p : CachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  let kv_cache := p.kv_cache
  let BLOCK_TABLES := p.BLOCK_TABLES
  let context_lengths := p.context_lengths
  triton {
    tl.region q = tl.float32, k = tl.float32, cos = tl.float32, sin = tl.float32, kv_cache = tl.float32, BLOCK_TABLES = tl.int64, context_lengths = tl.int64
    block_head_index = tl.program_id(0).to(tl.int64)
    if block_head_index < $((p.Q_HEAD_NUM : Int)) {
      block_token_index = tl.program_id(1).to(tl.int64)
      dim_range0 = tl.arange(0, $(p.BLOCK_PAIRS)).to(tl.int64)
      dim_range1 = dim_range0 + $(((p.HEAD_DIM / 2) : Int))
      off_q0 = block_token_index * $((p.q_token_stride : Int)) +
        block_head_index * $((p.q_head_stride : Int)) +
        dim_range0 * $((p.head_dim_stride : Int))
      off_q1 = off_q0 + $(((p.HEAD_DIM / 2) : Int)) * $((p.head_dim_stride : Int))
      off_k0 = block_token_index * $((p.k_token_stride : Int)) +
        block_head_index * $((p.k_head_stride : Int)) +
        dim_range0 * $((p.head_dim_stride : Int))
      off_k1 = block_token_index * $((p.k_token_stride : Int)) +
        block_head_index * $((p.k_head_stride : Int)) +
        dim_range1 * $((p.head_dim_stride : Int))
      loaded_q0 = tl.load(q + off_q0, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)), other=0.0)
      loaded_q1 = tl.load(q + off_q1, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)), other=0.0)
      loaded_k0 = tl.load(k + off_k0,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      loaded_k1 = tl.load(k + off_k1,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      off_cos_sin = block_token_index * $((p.cos_token_stride : Int)) + dim_range0 * $((p.cos_stride : Int))
      loaded_cos = tl.load(cos + off_cos_sin,
        mask=(block_token_index < $((p.q_total_tokens : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      off_sin = block_token_index * $((p.sin_token_stride : Int)) + dim_range0 * $((p.sin_stride : Int))
      loaded_sin = tl.load(sin + off_sin,
        mask=(block_token_index < $((p.q_total_tokens : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))),
        other=0.0)
      out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
      out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
      out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
      out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
      past_kv_seq_len = tl.load(context_lengths + block_token_index) - $((1 : Int))
      last_block_idx = past_kv_seq_len // $((p.block_size : Int))
      block_table_ptr = BLOCK_TABLES + block_token_index * $((p.bts_stride : Int))
      block_ids = tl.load(block_table_ptr +
        last_block_idx * $((p.btb_stride : Int)),
        mask=block_token_index < $((p.q_total_tokens : Int)))
      offsets_in_last_block = past_kv_seq_len % $((p.block_size : Int)) * $((p.cachebs_stride : Int))
      kv_range0 = block_ids * $((p.cacheb_stride : Int)) +
        block_head_index * $((p.cacheh_stride : Int)) +
        offsets_in_last_block +
        dim_range0 * $((p.cached_stride : Int))
      kv_range1 = block_ids * $((p.cacheb_stride : Int)) +
        block_head_index * $((p.cacheh_stride : Int)) +
        offsets_in_last_block +
        dim_range1 * $((p.cached_stride : Int))
      tl.store(kv_cache + kv_range0, out_k0,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))))
      tl.store(kv_cache + kv_range1, out_k1,
        mask=(block_head_index < $((p.K_HEAD_NUM : Int))) &
        (dim_range0 < $(((p.HEAD_DIM / 2) : Int))))
      tl.store(q + off_q0, out_q0, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)))
      tl.store(q + off_q1, out_q1, mask=dim_range0 < $(((p.HEAD_DIM / 2) : Int)))
    }
  }

end VeriTile.Bench.Repairs.RotaryEmbNopad.Kernels.LegacyPastedSurface
