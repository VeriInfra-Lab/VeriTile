import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Complete selected Python function bodies. Argument typing/binding remains
outside each quotation. The Python candidate is newly authored from the
existing shared-Q-offset Lean candidate; it is not a historical source. -/
namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Parameters where
  (x q_token_stride q_head_stride q_dim_stride k_token_stride k_head_stride k_dim_stride v_token_stride v_head_stride v_dim_stride cos_token_stride cos_stride sin_token_stride sin_stride length_stride kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM BLOCK_HALF BLOCK_DIM : Nat)

def callSignature : CallSignature where
  pointers := [("q",.fp32), ("k",.fp32), ("v",.fp32), ("cos",.fp32), ("sin",.fp32), ("k_cache",.fp32), ("v_cache",.fp32), ("BLOCK_TABLES",.i32), ("context_lengths",.i32)]
  scalars := [("x",.integer), ("q_token_stride",.integer), ("q_head_stride",.integer), ("q_dim_stride",.integer), ("k_token_stride",.integer), ("k_head_stride",.integer), ("k_dim_stride",.integer), ("v_token_stride",.integer), ("v_head_stride",.integer), ("v_dim_stride",.integer), ("cos_token_stride",.integer), ("cos_stride",.integer), ("sin_token_stride",.integer), ("sin_stride",.integer), ("length_stride",.integer), ("kcb_stride",.integer), ("kch_stride",.integer), ("kcsplit_x_stride",.integer), ("kcs_stride",.integer), ("kcd_stride",.integer), ("vcb_stride",.integer), ("vch_stride",.integer), ("vcs_stride",.integer), ("vcd_stride",.integer), ("bts_stride",.integer), ("btb_stride",.integer), ("block_size",.integer)]
  constexpr := ["KV_GROUP_NUM","HEAD_DIM","BLOCK_HALF","BLOCK_DIM"]

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
  let KV_GROUP_NUM := p.KV_GROUP_NUM
  let HEAD_DIM := p.HEAD_DIM
  let BLOCK_HALF := p.BLOCK_HALF
  let BLOCK_DIM := p.BLOCK_DIM
  let x : Int := p.x
  let q_token_stride : Int := p.q_token_stride
  let q_head_stride : Int := p.q_head_stride
  let q_dim_stride : Int := p.q_dim_stride
  let k_token_stride : Int := p.k_token_stride
  let k_head_stride : Int := p.k_head_stride
  let k_dim_stride : Int := p.k_dim_stride
  let v_token_stride : Int := p.v_token_stride
  let v_head_stride : Int := p.v_head_stride
  let v_dim_stride : Int := p.v_dim_stride
  let cos_token_stride : Int := p.cos_token_stride
  let cos_stride : Int := p.cos_stride
  let sin_token_stride : Int := p.sin_token_stride
  let sin_stride : Int := p.sin_stride
  let length_stride : Int := p.length_stride
  let kcb_stride : Int := p.kcb_stride
  let kch_stride : Int := p.kch_stride
  let kcsplit_x_stride : Int := p.kcsplit_x_stride
  let kcs_stride : Int := p.kcs_stride
  let kcd_stride : Int := p.kcd_stride
  let vcb_stride : Int := p.vcb_stride
  let vch_stride : Int := p.vch_stride
  let vcs_stride : Int := p.vcs_stride
  let vcd_stride : Int := p.vcd_stride
  let bts_stride : Int := p.bts_stride
  let btb_stride : Int := p.btb_stride
  let block_size : Int := p.block_size
  with_triton_call callSignature call_args [x := x, q_token_stride := q_token_stride, q_head_stride := q_head_stride, q_dim_stride := q_dim_stride, k_token_stride := k_token_stride, k_head_stride := k_head_stride, k_dim_stride := k_dim_stride, v_token_stride := v_token_stride, v_head_stride := v_head_stride, v_dim_stride := v_dim_stride, cos_token_stride := cos_token_stride, cos_stride := cos_stride, sin_token_stride := sin_token_stride, sin_stride := sin_stride, length_stride := length_stride, kcb_stride := kcb_stride, kch_stride := kch_stride, kcsplit_x_stride := kcsplit_x_stride, kcs_stride := kcs_stride, kcd_stride := kcd_stride, vcb_stride := vcb_stride, vch_stride := vch_stride, vcs_stride := vcs_stride, vcd_stride := vcd_stride, bts_stride := bts_stride, btb_stride := btb_stride, block_size := block_size] in triton {
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_idx = tl.program_id(1).to(tl.int64)
    dim_range = tl.arange(0, BLOCK_DIM).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_HALF).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2
    pair_mask = dim_range0 < HEAD_DIM // 2

    off_q = cur_token_idx * q_token_stride + cur_head_idx * q_head_stride
    off_q0 = off_q + dim_range0 * q_dim_stride
    off_q1 = off_q + dim_range1 * q_dim_stride
    loaded_q0 = tl.load(q + off_q0, mask=pair_mask, other=0)
    loaded_q1 = tl.load(q + off_q1, mask=pair_mask, other=0)
    off_cos = cur_token_idx * cos_token_stride + dim_range0 * cos_stride
    off_sin = cur_token_idx * sin_token_stride + dim_range0 * sin_stride
    loaded_cos = tl.load(cos + off_cos, mask=pair_mask, other=0)
    loaded_sin = tl.load(sin + off_sin, mask=pair_mask, other=0)
    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0, mask=pair_mask)
    tl.store(q + off_q1, out_q1, mask=pair_mask)

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        cur_k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k = cur_token_idx * k_token_stride + cur_k_head_idx * k_head_stride
        off_k0 = off_k + dim_range0 * k_dim_stride
        off_k1 = off_k + dim_range1 * k_dim_stride
        loaded_k0 = tl.load(k + off_k0, mask=pair_mask, other=0)
        loaded_k1 = tl.load(k + off_k1, mask=pair_mask, other=0)
        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx * length_stride).to(tl.int64) - 1
        last_block_idx = past_kv_seq_len // block_size
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * bts_stride + last_block_idx * btb_stride).to(tl.int64)
        offsets_in_last_block = past_kv_seq_len % block_size
        offsets_cache_base = block_ids * kcb_stride + cur_k_head_idx * kch_stride
        k_range0 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range0 // x) * kcsplit_x_stride
                    + (dim_range0 % x) * kcd_stride)
        k_range1 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range1 // x) * kcsplit_x_stride
                    + (dim_range1 % x) * kcd_stride)
        tl.store(k_cache + k_range0, out_k0, mask=pair_mask)
        tl.store(k_cache + k_range1, out_k1, mask=pair_mask)
        off_v = (cur_token_idx * v_token_stride + cur_k_head_idx * v_head_stride
                 + dim_range * v_dim_stride)
        loaded_v = tl.load(v + off_v, mask=dim_range < HEAD_DIM, other=0)
        v_range = (block_ids * vcb_stride + cur_k_head_idx * vch_stride
                   + offsets_in_last_block * vcs_stride + dim_range * vcd_stride)
        tl.store(v_cache + v_range, loaded_v, mask=dim_range < HEAD_DIM)
  }

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
  let KV_GROUP_NUM := p.KV_GROUP_NUM
  let HEAD_DIM := p.HEAD_DIM
  let BLOCK_HALF := p.BLOCK_HALF
  let BLOCK_DIM := p.BLOCK_DIM
  let x : Int := p.x
  let q_token_stride : Int := p.q_token_stride
  let q_head_stride : Int := p.q_head_stride
  let q_dim_stride : Int := p.q_dim_stride
  let k_token_stride : Int := p.k_token_stride
  let k_head_stride : Int := p.k_head_stride
  let k_dim_stride : Int := p.k_dim_stride
  let v_token_stride : Int := p.v_token_stride
  let v_head_stride : Int := p.v_head_stride
  let v_dim_stride : Int := p.v_dim_stride
  let cos_token_stride : Int := p.cos_token_stride
  let cos_stride : Int := p.cos_stride
  let sin_token_stride : Int := p.sin_token_stride
  let sin_stride : Int := p.sin_stride
  let length_stride : Int := p.length_stride
  let kcb_stride : Int := p.kcb_stride
  let kch_stride : Int := p.kch_stride
  let kcsplit_x_stride : Int := p.kcsplit_x_stride
  let kcs_stride : Int := p.kcs_stride
  let kcd_stride : Int := p.kcd_stride
  let vcb_stride : Int := p.vcb_stride
  let vch_stride : Int := p.vch_stride
  let vcs_stride : Int := p.vcs_stride
  let vcd_stride : Int := p.vcd_stride
  let bts_stride : Int := p.bts_stride
  let btb_stride : Int := p.btb_stride
  let block_size : Int := p.block_size
  with_triton_call callSignature call_args [x := x, q_token_stride := q_token_stride, q_head_stride := q_head_stride, q_dim_stride := q_dim_stride, k_token_stride := k_token_stride, k_head_stride := k_head_stride, k_dim_stride := k_dim_stride, v_token_stride := v_token_stride, v_head_stride := v_head_stride, v_dim_stride := v_dim_stride, cos_token_stride := cos_token_stride, cos_stride := cos_stride, sin_token_stride := sin_token_stride, sin_stride := sin_stride, length_stride := length_stride, kcb_stride := kcb_stride, kch_stride := kch_stride, kcsplit_x_stride := kcsplit_x_stride, kcs_stride := kcs_stride, kcd_stride := kcd_stride, vcb_stride := vcb_stride, vch_stride := vch_stride, vcs_stride := vcs_stride, vcd_stride := vcd_stride, bts_stride := bts_stride, btb_stride := btb_stride, block_size := block_size] in triton {
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_idx = tl.program_id(1).to(tl.int64)
    dim_range = tl.arange(0, BLOCK_DIM).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_HALF).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2
    pair_mask = dim_range0 < HEAD_DIM // 2

    off_q = cur_token_idx * q_token_stride + cur_head_idx * q_head_stride
    off_q0 = off_q + dim_range0 * q_dim_stride
    off_q1 = off_q0 + (HEAD_DIM // 2) * q_dim_stride
    loaded_q0 = tl.load(q + off_q0, mask=pair_mask, other=0)
    loaded_q1 = tl.load(q + off_q1, mask=pair_mask, other=0)
    off_cos = cur_token_idx * cos_token_stride + dim_range0 * cos_stride
    off_sin = cur_token_idx * sin_token_stride + dim_range0 * sin_stride
    loaded_cos = tl.load(cos + off_cos, mask=pair_mask, other=0)
    loaded_sin = tl.load(sin + off_sin, mask=pair_mask, other=0)
    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0, mask=pair_mask)
    tl.store(q + off_q1, out_q1, mask=pair_mask)

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        cur_k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k = cur_token_idx * k_token_stride + cur_k_head_idx * k_head_stride
        off_k0 = off_k + dim_range0 * k_dim_stride
        off_k1 = off_k + dim_range1 * k_dim_stride
        loaded_k0 = tl.load(k + off_k0, mask=pair_mask, other=0)
        loaded_k1 = tl.load(k + off_k1, mask=pair_mask, other=0)
        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx * length_stride).to(tl.int64) - 1
        last_block_idx = past_kv_seq_len // block_size
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * bts_stride + last_block_idx * btb_stride).to(tl.int64)
        offsets_in_last_block = past_kv_seq_len % block_size
        offsets_cache_base = block_ids * kcb_stride + cur_k_head_idx * kch_stride
        k_range0 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range0 // x) * kcsplit_x_stride
                    + (dim_range0 % x) * kcd_stride)
        k_range1 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range1 // x) * kcsplit_x_stride
                    + (dim_range1 % x) * kcd_stride)
        tl.store(k_cache + k_range0, out_k0, mask=pair_mask)
        tl.store(k_cache + k_range1, out_k1, mask=pair_mask)
        off_v = (cur_token_idx * v_token_stride + cur_k_head_idx * v_head_stride
                 + dim_range * v_dim_stride)
        loaded_v = tl.load(v + off_v, mask=dim_range < HEAD_DIM, other=0)
        v_range = (block_ids * vcb_stride + cur_k_head_idx * vch_stride
                   + offsets_in_last_block * vcs_stride + dim_range * vcd_stride)
        tl.store(v_cache + v_range, loaded_v, mask=dim_range < HEAD_DIM)
  }

end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
