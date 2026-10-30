import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.MultinomialSampling.Source
open VeriTile.Triton

/-! Complete direct sources. FP32 is selected; the other existing storage
profiles are retained for historical helpers. Constexpr bindings preserve the
checked arange shapes; signed storage, RNG, masks and label clamp remain. -/
def original
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel := triton {
    tl.region Scores = tl.float32, Seeds = tl.int64, Offsets = tl.int64,
      Indices = tl.int64, Outputs = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)
    BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hN)
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK_arg + tl.arange(0, BLOCK_arg)
    n_off = tl.arange(0, BLOCK_N_arg)
    off_mask = off < $(num_batchs)
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros([$(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)], dtype=tl.float32)
    output = tl.load(Indices + off * $(stride_ib), mask=off_mask)
    for b_idx in range($(0), $(num_tokens), $(BLOCK_N)) {
      s_off = b_idx + n_off
      s_mask = off_mask[:, None] & (s_off[None, :] < $(num_tokens))
      scores = tl.load(Scores + off[:, None] * $(stride_sb) + s_off[None, :] * $(stride_st),
        mask=s_mask, other=0.0).to(tl.float32)
      c_scores = tl.cumsum(scores, 1)
      cum_scores = acc[:, None] + c_scores
      acc += tl.max(c_scores, 1)
      pre_cum_scores = cum_scores - scores
      valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
      found_mask = tl.sum(valid_mask, 1) > 0
      valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
      indices = tl.load(Indices + off * $(stride_ib) + tl.minimum(valid_pos, $(num_tokens-1)) * $(stride_it),
        mask=found_mask & off_mask, other=-1)
      output = tl.where(found_mask, indices, output)
    }
    tl.store(Outputs + off, output, mask=off_mask)
  }

def original16
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel := triton {
    tl.region Scores = tl.float16, Seeds = tl.int64, Offsets = tl.int64,
      Indices = tl.int64, Outputs = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)
    BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hN)
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK_arg + tl.arange(0, BLOCK_arg)
    n_off = tl.arange(0, BLOCK_N_arg)
    off_mask = off < $(num_batchs)
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros([$(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)], dtype=tl.float32)
    output = tl.load(Indices + off * $(stride_ib), mask=off_mask)
    for b_idx in range($(0), $(num_tokens), $(BLOCK_N)) {
      s_off = b_idx + n_off
      s_mask = off_mask[:, None] & (s_off[None, :] < $(num_tokens))
      scores = tl.load(Scores + off[:, None] * $(stride_sb) + s_off[None, :] * $(stride_st),
        mask=s_mask, other=0.0).to(tl.float32)
      c_scores = tl.cumsum(scores, 1)
      cum_scores = acc[:, None] + c_scores
      acc += tl.max(c_scores, 1)
      pre_cum_scores = cum_scores - scores
      valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
      found_mask = tl.sum(valid_mask, 1) > 0
      valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
      indices = tl.load(Indices + off * $(stride_ib) + tl.minimum(valid_pos, $(num_tokens-1)) * $(stride_it),
        mask=found_mask & off_mask, other=-1)
      output = tl.where(found_mask, indices, output)
    }
    tl.store(Outputs + off, output, mask=off_mask)
  }

def originalBfloat16
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel := triton {
    tl.region Scores = tl.bfloat16, Seeds = tl.int64, Offsets = tl.int64,
      Indices = tl.int64, Outputs = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)
    BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hN)
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK_arg + tl.arange(0, BLOCK_arg)
    n_off = tl.arange(0, BLOCK_N_arg)
    off_mask = off < $(num_batchs)
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros([$(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)], dtype=tl.float32)
    output = tl.load(Indices + off * $(stride_ib), mask=off_mask)
    for b_idx in range($(0), $(num_tokens), $(BLOCK_N)) {
      s_off = b_idx + n_off
      s_mask = off_mask[:, None] & (s_off[None, :] < $(num_tokens))
      scores = tl.load(Scores + off[:, None] * $(stride_sb) + s_off[None, :] * $(stride_st),
        mask=s_mask, other=0.0).to(tl.float32)
      c_scores = tl.cumsum(scores, 1)
      cum_scores = acc[:, None] + c_scores
      acc += tl.max(c_scores, 1)
      pre_cum_scores = cum_scores - scores
      valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
      found_mask = tl.sum(valid_mask, 1) > 0
      valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
      indices = tl.load(Indices + off * $(stride_ib) + tl.minimum(valid_pos, $(num_tokens-1)) * $(stride_it),
        mask=found_mask & off_mask, other=-1)
      output = tl.where(found_mask, indices, output)
    }
    tl.store(Outputs + off, output, mask=off_mask)
  }

def original64
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel := triton {
    tl.region Scores = tl.float64, Seeds = tl.int64, Offsets = tl.int64,
      Indices = tl.int64, Outputs = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)
    BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hN)
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK_arg + tl.arange(0, BLOCK_arg)
    n_off = tl.arange(0, BLOCK_N_arg)
    off_mask = off < $(num_batchs)
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros([$(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)], dtype=tl.float32)
    output = tl.load(Indices + off * $(stride_ib), mask=off_mask)
    for b_idx in range($(0), $(num_tokens), $(BLOCK_N)) {
      s_off = b_idx + n_off
      s_mask = off_mask[:, None] & (s_off[None, :] < $(num_tokens))
      scores = tl.load(Scores + off[:, None] * $(stride_sb) + s_off[None, :] * $(stride_st),
        mask=s_mask, other=0.0).to(tl.float32)
      c_scores = tl.cumsum(scores, 1)
      cum_scores = acc[:, None] + c_scores
      acc += tl.max(c_scores, 1)
      pre_cum_scores = cum_scores - scores
      valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
      found_mask = tl.sum(valid_mask, 1) > 0
      valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
      indices = tl.load(Indices + off * $(stride_ib) + tl.minimum(valid_pos, $(num_tokens-1)) * $(stride_it),
        mask=found_mask & off_mask, other=-1)
      output = tl.where(found_mask, indices, output)
    }
    tl.store(Outputs + off, output, mask=off_mask)
  }

/-- Specialize and statically unroll the complete token-block loop. -/
def unrolled
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel := triton {
    tl.region Scores = tl.float32, Seeds = tl.int64, Offsets = tl.int64,
      Indices = tl.int64, Outputs = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)
    BLOCK_N_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_N hN)
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK_arg + tl.arange(0, BLOCK_arg)
    n_off = tl.arange(0, BLOCK_N_arg)
    off_mask = off < $(num_batchs)
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros([$(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)], dtype=tl.float32)
    output = tl.load(Indices + off * $(stride_ib), mask=off_mask)
    for b_idx in tl.static_range($((num_tokens + BLOCK_N - 1) / BLOCK_N)) {
      b_idx = b_idx * $(BLOCK_N)
      s_off = b_idx + n_off
      s_mask = off_mask[:, None] & (s_off[None, :] < $(num_tokens))
      scores = tl.load(Scores + off[:, None] * $(stride_sb) + s_off[None, :] * $(stride_st),
        mask=s_mask, other=0.0).to(tl.float32)
      c_scores = tl.cumsum(scores, 1)
      cum_scores = acc[:, None] + c_scores
      acc += tl.max(c_scores, 1)
      pre_cum_scores = cum_scores - scores
      valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
      found_mask = tl.sum(valid_mask, 1) > 0
      valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
      indices = tl.load(Indices + off * $(stride_ib) + tl.minimum(valid_pos, $(num_tokens-1)) * $(stride_it),
        mask=found_mask & off_mask, other=-1)
      output = tl.where(found_mask, indices, output)
    }
    tl.store(Outputs + off, output, mask=off_mask)
  }

end VeriTile.Bench.Repairs.MultinomialSampling.Source
