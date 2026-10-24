import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Parameters where
  (x q_token_stride q_head_stride q_dim_stride k_token_stride k_head_stride k_dim_stride v_token_stride v_head_stride v_dim_stride cos_token_stride cos_stride sin_token_stride sin_stride length_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM BLOCK_HALF BLOCK_DIM : Nat)

/-- Complete unchanged decoding kernel: Q stores, conditional K-cache scatter,
full-width V copy, independent coefficient pitches and both cache layouts. -/
def kernel (p : Parameters) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let cos : RegionName := "cos"
  let sin : RegionName := "sin"
  let k_cache : RegionName := "k_cache"
  let v_cache : RegionName := "v_cache"
  let BLOCK_TABLES : RegionName := "BLOCK_TABLES"
  let context_lengths : RegionName := "context_lengths"
  triton {
    tl.region q = tl.float32, k = tl.float32, v = tl.float32, cos = tl.float32, sin = tl.float32, k_cache = tl.float32, v_cache = tl.float32, BLOCK_TABLES = tl.int32, context_lengths = tl.int32
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_idx = tl.program_id(1).to(tl.int64)
    dim_range = tl.arange(0, $(p.BLOCK_DIM)).to(tl.int64)
    dim_range0 = tl.arange(0, $(p.BLOCK_HALF)).to(tl.int64)
    dim_range1 = dim_range0 + $((p.HEAD_DIM : Int)) // $((2 : Int))
    pair_mask = dim_range0 < $((p.HEAD_DIM : Int)) // $((2 : Int))
    off_q = cur_token_idx * $((p.q_token_stride : Int)) + cur_head_idx * $((p.q_head_stride : Int))
    off_q0 = off_q + dim_range0 * $((p.q_dim_stride : Int))
    off_q1 = off_q + dim_range1 * $((p.q_dim_stride : Int))
    loaded_q0 = tl.load(q + off_q0, mask=pair_mask, other=0)
    loaded_q1 = tl.load(q + off_q1, mask=pair_mask, other=0)
    off_cos = cur_token_idx * $((p.cos_token_stride : Int)) + dim_range0 * $((p.cos_stride : Int))
    off_sin = cur_token_idx * $((p.sin_token_stride : Int)) + dim_range0 * $((p.sin_stride : Int))
    loaded_cos = tl.load(cos + off_cos, mask=pair_mask, other=0)
    loaded_sin = tl.load(sin + off_sin, mask=pair_mask, other=0)
    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0, mask=pair_mask)
    tl.store(q + off_q1, out_q1, mask=pair_mask)
    handle_kv = cur_head_idx % $((p.KV_GROUP_NUM : Int)) == $((0 : Int))
    if handle_kv {
        cur_k_head_idx = cur_head_idx // $((p.KV_GROUP_NUM : Int))
        off_k = cur_token_idx * $((p.k_token_stride : Int)) + cur_k_head_idx * $((p.k_head_stride : Int))
        off_k0 = off_k + dim_range0 * $((p.k_dim_stride : Int))
        off_k1 = off_k + dim_range1 * $((p.k_dim_stride : Int))
        loaded_k0 = tl.load(k + off_k0, mask=pair_mask, other=0)
        loaded_k1 = tl.load(k + off_k1, mask=pair_mask, other=0)
        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx * $((p.length_stride : Int))).to(tl.int64) - $((1 : Int))
        last_block_idx = past_kv_seq_len // $((p.block_size : Int))
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $((p.bts_stride : Int)) + last_block_idx * $((p.btb_stride : Int))).to(tl.int64)
        offsets_in_last_block = past_kv_seq_len % $((p.block_size : Int))
        offsets_cache_base = block_ids * $((p.kcb_stride : Int)) + cur_k_head_idx * $((p.kch_stride : Int))
        k_range0 = offsets_cache_base + offsets_in_last_block * $((p.kcs_stride : Int)) + dim_range0 // $((p.x : Int)) * $((p.kcsplit_x_stride : Int)) + dim_range0 % $((p.x : Int)) * $((p.kcd_stride : Int))
        k_range1 = offsets_cache_base + offsets_in_last_block * $((p.kcs_stride : Int)) + dim_range1 // $((p.x : Int)) * $((p.kcsplit_x_stride : Int)) + dim_range1 % $((p.x : Int)) * $((p.kcd_stride : Int))
        tl.store(k_cache + k_range0, out_k0, mask=pair_mask)
        tl.store(k_cache + k_range1, out_k1, mask=pair_mask)
        off_v = cur_token_idx * $((p.v_token_stride : Int)) + cur_k_head_idx * $((p.v_head_stride : Int)) + dim_range * $((p.v_dim_stride : Int))
        loaded_v = tl.load(v + off_v, mask=dim_range < $((p.HEAD_DIM : Int)), other=0)
        v_range = block_ids * $((p.vcb_stride : Int)) + cur_k_head_idx * $((p.vch_stride : Int)) + offsets_in_last_block * $((p.vcs_stride : Int)) + dim_range * $((p.vcd_stride : Int))
        tl.store(v_cache + v_range, loaded_v, mask=dim_range < $((p.HEAD_DIM : Int)))
    }
  }

/-- Reuse the first-half Q address and a constexpr half-stride displacement,
removing the second per-lane Q stride multiplication. All K/V work is intact. -/
def kernelSharedQOffset (p : Parameters) : ComputeKernel :=
  let q : RegionName := "q"
  let k : RegionName := "k"
  let v : RegionName := "v"
  let cos : RegionName := "cos"
  let sin : RegionName := "sin"
  let k_cache : RegionName := "k_cache"
  let v_cache : RegionName := "v_cache"
  let BLOCK_TABLES : RegionName := "BLOCK_TABLES"
  let context_lengths : RegionName := "context_lengths"
  triton {
    tl.region q = tl.float32, k = tl.float32, v = tl.float32, cos = tl.float32, sin = tl.float32, k_cache = tl.float32, v_cache = tl.float32, BLOCK_TABLES = tl.int32, context_lengths = tl.int32
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_idx = tl.program_id(1).to(tl.int64)
    dim_range = tl.arange(0, $(p.BLOCK_DIM)).to(tl.int64)
    dim_range0 = tl.arange(0, $(p.BLOCK_HALF)).to(tl.int64)
    dim_range1 = dim_range0 + $((p.HEAD_DIM : Int)) // $((2 : Int))
    pair_mask = dim_range0 < $((p.HEAD_DIM : Int)) // $((2 : Int))
    off_q = cur_token_idx * $((p.q_token_stride : Int)) + cur_head_idx * $((p.q_head_stride : Int))
    off_q0 = off_q + dim_range0 * $((p.q_dim_stride : Int))
    off_q1 = off_q0 + $((((p.HEAD_DIM : Int) / 2) * (p.q_dim_stride : Int) : Int))
    loaded_q0 = tl.load(q + off_q0, mask=pair_mask, other=0)
    loaded_q1 = tl.load(q + off_q1, mask=pair_mask, other=0)
    off_cos = cur_token_idx * $((p.cos_token_stride : Int)) + dim_range0 * $((p.cos_stride : Int))
    off_sin = cur_token_idx * $((p.sin_token_stride : Int)) + dim_range0 * $((p.sin_stride : Int))
    loaded_cos = tl.load(cos + off_cos, mask=pair_mask, other=0)
    loaded_sin = tl.load(sin + off_sin, mask=pair_mask, other=0)
    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0, mask=pair_mask)
    tl.store(q + off_q1, out_q1, mask=pair_mask)
    handle_kv = cur_head_idx % $((p.KV_GROUP_NUM : Int)) == $((0 : Int))
    if handle_kv {
        cur_k_head_idx = cur_head_idx // $((p.KV_GROUP_NUM : Int))
        off_k = cur_token_idx * $((p.k_token_stride : Int)) + cur_k_head_idx * $((p.k_head_stride : Int))
        off_k0 = off_k + dim_range0 * $((p.k_dim_stride : Int))
        off_k1 = off_k + dim_range1 * $((p.k_dim_stride : Int))
        loaded_k0 = tl.load(k + off_k0, mask=pair_mask, other=0)
        loaded_k1 = tl.load(k + off_k1, mask=pair_mask, other=0)
        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx * $((p.length_stride : Int))).to(tl.int64) - $((1 : Int))
        last_block_idx = past_kv_seq_len // $((p.block_size : Int))
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $((p.bts_stride : Int)) + last_block_idx * $((p.btb_stride : Int))).to(tl.int64)
        offsets_in_last_block = past_kv_seq_len % $((p.block_size : Int))
        offsets_cache_base = block_ids * $((p.kcb_stride : Int)) + cur_k_head_idx * $((p.kch_stride : Int))
        k_range0 = offsets_cache_base + offsets_in_last_block * $((p.kcs_stride : Int)) + dim_range0 // $((p.x : Int)) * $((p.kcsplit_x_stride : Int)) + dim_range0 % $((p.x : Int)) * $((p.kcd_stride : Int))
        k_range1 = offsets_cache_base + offsets_in_last_block * $((p.kcs_stride : Int)) + dim_range1 // $((p.x : Int)) * $((p.kcsplit_x_stride : Int)) + dim_range1 % $((p.x : Int)) * $((p.kcd_stride : Int))
        tl.store(k_cache + k_range0, out_k0, mask=pair_mask)
        tl.store(k_cache + k_range1, out_k1, mask=pair_mask)
        off_v = cur_token_idx * $((p.v_token_stride : Int)) + cur_k_head_idx * $((p.v_head_stride : Int)) + dim_range * $((p.v_dim_stride : Int))
        loaded_v = tl.load(v + off_v, mask=dim_range < $((p.HEAD_DIM : Int)), other=0)
        v_range = block_ids * $((p.vcb_stride : Int)) + cur_k_head_idx * $((p.vch_stride : Int)) + offsets_in_last_block * $((p.vcs_stride : Int)) + dim_range * $((p.vcd_stride : Int))
        tl.store(v_cache + v_range, loaded_v, mask=dim_range < $((p.HEAD_DIM : Int)))
    }
  }

end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
