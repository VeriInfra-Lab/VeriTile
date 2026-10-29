import VeriTile.Triton.DSL.CallSignature

/-! Complete original Python bodies. Caller pointer types, scalar arguments
and checked arange sizes stay outside the quotations. FP32 is the selected
Python contract; the other storage instances retain historical proof scope. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.Source
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000

def originalSignature : CallSignature where
  pointers := [("Scores",.fp32),("Seeds",.i64),("Offsets",.i64),("Indices",.i64),("Outputs",.i64)]
  scalars := [("stride_sb",.i32), ("stride_st",.i32), ("stride_ib",.i32), ("stride_it",.i32), ("num_batchs",.i32), ("num_tokens",.i32)]
  constexpr := ["BLOCK", "BLOCK_N"]
  readPorts := ["Scores","Seeds","Offsets","Indices"]
  writePorts := ["Outputs"]

def original
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel :=
  with_triton_call originalSignature call_args [stride_sb := stride_sb,
    stride_st := stride_st, stride_ib := stride_ib, stride_it := stride_it,
    num_batchs := num_batchs, num_tokens := num_tokens,
    BLOCK := Random.i32ArangeSize BLOCK hB,
    BLOCK_N := Random.i32ArangeSize BLOCK_N hN] in
  triton {
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)
  }

def original16Signature : CallSignature where
  pointers := [("Scores",.fp16),("Seeds",.i64),("Offsets",.i64),("Indices",.i64),("Outputs",.i64)]
  scalars := [("stride_sb",.i32), ("stride_st",.i32), ("stride_ib",.i32), ("stride_it",.i32), ("num_batchs",.i32), ("num_tokens",.i32)]
  constexpr := ["BLOCK", "BLOCK_N"]
  readPorts := ["Scores","Seeds","Offsets","Indices"]
  writePorts := ["Outputs"]

def original16
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel :=
  with_triton_call original16Signature call_args [stride_sb := stride_sb,
    stride_st := stride_st, stride_ib := stride_ib, stride_it := stride_it,
    num_batchs := num_batchs, num_tokens := num_tokens,
    BLOCK := Random.i32ArangeSize BLOCK hB,
    BLOCK_N := Random.i32ArangeSize BLOCK_N hN] in
  triton {
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)
  }

def originalBfloat16Signature : CallSignature where
  pointers := [("Scores",.bf16),("Seeds",.i64),("Offsets",.i64),("Indices",.i64),("Outputs",.i64)]
  scalars := [("stride_sb",.i32), ("stride_st",.i32), ("stride_ib",.i32), ("stride_it",.i32), ("num_batchs",.i32), ("num_tokens",.i32)]
  constexpr := ["BLOCK", "BLOCK_N"]
  readPorts := ["Scores","Seeds","Offsets","Indices"]
  writePorts := ["Outputs"]

def originalBfloat16
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel :=
  with_triton_call originalBfloat16Signature call_args [stride_sb := stride_sb,
    stride_st := stride_st, stride_ib := stride_ib, stride_it := stride_it,
    num_batchs := num_batchs, num_tokens := num_tokens,
    BLOCK := Random.i32ArangeSize BLOCK hB,
    BLOCK_N := Random.i32ArangeSize BLOCK_N hN] in
  triton {
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)
  }

def original64Signature : CallSignature where
  pointers := [("Scores",.fp64),("Seeds",.i64),("Offsets",.i64),("Indices",.i64),("Outputs",.i64)]
  scalars := [("stride_sb",.i32), ("stride_st",.i32), ("stride_ib",.i32), ("stride_it",.i32), ("num_batchs",.i32), ("num_tokens",.i32)]
  constexpr := ["BLOCK", "BLOCK_N"]
  readPorts := ["Scores","Seeds","Offsets","Indices"]
  writePorts := ["Outputs"]

def original64
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel :=
  with_triton_call original64Signature call_args [stride_sb := stride_sb,
    stride_st := stride_st, stride_ib := stride_ib, stride_it := stride_it,
    num_batchs := num_batchs, num_tokens := num_tokens,
    BLOCK := Random.i32ArangeSize BLOCK hB,
    BLOCK_N := Random.i32ArangeSize BLOCK_N hN] in
  triton {
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)
  }

def unrolledSignature : CallSignature where
  pointers := [("Scores",.fp32),("Seeds",.i64),("Offsets",.i64),("Indices",.i64),("Outputs",.i64)]
  scalars := [("stride_sb",.i32), ("stride_st",.i32), ("stride_ib",.i32), ("stride_it",.i32), ("num_batchs",.i32)]
  constexpr := ["num_tokens", "BLOCK", "BLOCK_N"]
  readPorts := ["Scores","Seeds","Offsets","Indices"]
  writePorts := ["Outputs"]

def unrolled
    (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) :
    ComputeKernel :=
  with_triton_call unrolledSignature call_args [stride_sb := stride_sb,
    stride_st := stride_st, stride_ib := stride_ib, stride_it := stride_it,
    num_batchs := num_batchs, num_tokens := num_tokens,
    BLOCK := Random.i32ArangeSize BLOCK hB,
    BLOCK_N := Random.i32ArangeSize BLOCK_N hN] in
  triton {
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in tl.static_range(tl.cdiv(num_tokens, BLOCK_N)):
        b_idx = b_idx * BLOCK_N
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)
  }

end
end VeriTile.Bench.Repairs.MultinomialSampling.Source
