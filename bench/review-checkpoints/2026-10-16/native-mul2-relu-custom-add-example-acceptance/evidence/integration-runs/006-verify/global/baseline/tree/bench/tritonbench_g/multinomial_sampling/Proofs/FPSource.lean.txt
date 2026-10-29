import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.SignedCast

namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
open VeriTile.Triton Lean

/-! Full original kernel, copied from the frozen four-storage translation probe.
The source shape module checks equality to that probe. Two checked constexpr
bindings and the equivalent repeated zero shape preserve the original block
sizes. I64 metadata, explicit signed32 narrowing, concrete RNG operands,
masked loads, scans, reductions, the entire raw range and final store remain.
This is a task-local source port, not a general Python/compiler correctness
claim; native address/ABI and numerical refinement are separate obligations. -/
syntax "multinomialSource%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(multinomialSource%[$storage:tritonDType]) =>
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

def original := multinomialSource%[tl.float32]
def original16 := multinomialSource%[tl.float16]
def originalBfloat16 := multinomialSource%[tl.bfloat16]
def original64 := multinomialSource%[tl.float64]

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

def Storage.dtype : Storage → FloatDType
  | .fp32 => .real | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .real

structure Parameters where
  scores : RegionName
  seeds : RegionName
  offsets : RegionName
  indices : RegionName
  output : RegionName
  strideScoreBatch : Nat
  strideScoreToken : Nat
  strideIndexBatch : Nat
  strideIndexToken : Nat
  batches : Nat
  tokens : Nat
  block : Nat
  blockTokens : Nat
  blockDomain : Random.ArangeSizeDomain block
  tokensDomain : Random.ArangeSizeDomain blockTokens

def source (d : Storage) (p : Parameters) : ComputeKernel :=
  let kernel := match d with
    | .fp32 => original | .fp16 => original16 | .bf16 => originalBfloat16 | .fp64 => original64
  kernel p.scores p.seeds p.offsets p.indices p.output
    p.strideScoreBatch p.strideScoreToken p.strideIndexBatch p.strideIndexToken
    p.batches p.tokens p.block p.blockTokens p.blockDomain p.tokensDomain

def prologue (d : Storage) (p : Parameters) : List ComputeStmt :=
  (source d p).surfaceBody.take 11

def body (d : Storage) (p : Parameters) : List ComputeStmt :=
  match ((source d p).surfaceBody.drop 11).head? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []

def epilogue (d : Storage) (p : Parameters) : List ComputeStmt :=
  (source d p).surfaceBody.drop 12

theorem source_decomposition (d : Storage) (p : Parameters) :
    (source d p).surfaceBody = prologue d p ++
      [.forRange "b_idx" 0 p.tokens p.blockTokens (body d p)] ++ epilogue d p := by
  cases d <;> rfl

#axiomsClean source_decomposition
#axiomsClean source
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
