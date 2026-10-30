import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FPMultinomialSource
open VeriTile.Triton Lean

/-! Full-body frontend regression for TritonBench@603e28a multinomial_sampling.py
(SHA256 3bd5a7c11f42ade4e2ae69b58019888a419990bfbf3fe1d54d877e50e36de336).
Four Score storage profiles retain original i64 seed/offset/index/output storage, signed
narrowing before RNG, masked loads, cumsum, Bool sum and argmax. Two explicit
checked identity bindings materialize the existing constexprs. Zeros repeats
the same checked dimension. This is elaboration/axiom evidence, not the ordinary
corpus source adapter, a paired optimization proof or an execution/value theorem. -/
syntax "multinomialFrontend%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(multinomialFrontend%[$storage:tritonDType]) =>
    `(fun (Scores Seeds Offsets Indices Outputs : RegionName)
    (stride_sb stride_st stride_ib stride_it num_batchs num_tokens BLOCK BLOCK_N : Nat)
    (hB : Random.ArangeSizeDomain BLOCK) (hN : Random.ArangeSizeDomain BLOCK_N) =>
  ComputeKernel.mk [Scores, Seeds, Offsets, Indices] [Outputs] (triton {
    tl.region Scores = $storage, Seeds = tl.int64, Offsets = tl.int64,
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
      indices = tl.load(Indices + off * $(stride_ib) + valid_pos * $(stride_it),
        mask=found_mask & off_mask, other=-1)
      output = tl.where(found_mask, indices, output)
    }
    tl.store(Outputs + off, output, mask=off_mask)
  }).surfaceBody)

def original := multinomialFrontend%[tl.float32]
def original16 := multinomialFrontend%[tl.float16]
def originalBfloat16 := multinomialFrontend%[tl.bfloat16]
def original64 := multinomialFrontend%[tl.float64]

#axiomsClean original
#axiomsClean original16
#axiomsClean originalBfloat16
#axiomsClean original64
end VeriTile.Tests.FPMultinomialSource
