import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! FP32 payloads and signed int32 ID storage. Widened source indices are
interpreted exactly here; Proofs/Native checks their fixed-width bounds. -/
namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel
open VeriTile.Triton

inductive IndexType where | int32
  deriving DecidableEq

structure Parameters where
  (weight ids output : RegionName)
  (first last : Int)
  (weightStride outputStride n hidden width : Nat)

def source : IndexType → Parameters → ComputeKernel
  | .int32, p =>
    let weight := p.weight
    let input_ids := p.ids
    let out := p.output
    triton {
  tl.ports inputs=[weight,input_ids], outputs=[out]
      tl.region weight = tl.float32, input_ids = tl.int32, out = tl.float32
      start_n = tl.program_id(0) * $(64)
      offs_nn = start_n + tl.arange(0, $(1))
      offs_d = tl.arange(0, $(p.width))
      for start_nn in range(0, $(64), $(1)) {
        start_nn = tl.multiple_of(start_nn, $(1))
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < $(p.n)
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=$((0 : Int)))
        id_mask = n_ctx_mask & (token_ids >= $((p.first : Int))) & (token_ids < $((p.last : Int)))
        token_ids = tl.where(id_mask, token_ids, $((p.first : Int))) - $((p.first : Int))
        dim_mask = offs_d < $(p.hidden)
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * $(p.weightStride) + offs_d[None, :], mask=load_mask, other=0.0)
        output_rows = tl.where(n_ctx_mask, offs_seq, $(0))
        tl.store(out + output_rows[:, None] * $(p.outputStride) + offs_d[None, :], vecs, mask=store_mask)
      }
    }

/-- Existing integer-index-commuted candidate; all loads, masks and stores retained. -/
def indexCommutedSource : IndexType → Parameters → ComputeKernel
  | .int32, p =>
    let weight := p.weight
    let input_ids := p.ids
    let out := p.output
    triton {
  tl.ports inputs=[weight,input_ids], outputs=[out]
      tl.region weight = tl.float32, input_ids = tl.int32, out = tl.float32
      start_n = $(64) * tl.program_id(0)
      offs_nn = start_n + tl.arange(0, $(1))
      offs_d = tl.arange(0, $(p.width))
      for start_nn in range(0, $(64), $(1)) {
        start_nn = tl.multiple_of(start_nn, $(1))
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < $(p.n)
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=$((0 : Int)))
        id_mask = n_ctx_mask & (token_ids >= $((p.first : Int))) & (token_ids < $((p.last : Int)))
        token_ids = tl.where(id_mask, token_ids, $((p.first : Int))) - $((p.first : Int))
        dim_mask = offs_d < $(p.hidden)
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * $(p.weightStride) + offs_d[None, :], mask=load_mask, other=0.0)
        output_rows = tl.where(n_ctx_mask, offs_seq, $(0))
        tl.store(out + output_rows[:, None] * $(p.outputStride) + offs_d[None, :], vecs, mask=store_mask)
      }
    }

def grid (n : Nat) : Nat := (n+63)/64

end VeriTile.Bench.Repairs.EmbeddingTritonKernel
