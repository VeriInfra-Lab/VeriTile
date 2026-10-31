import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Unchanged raw selected bodies with source signed64 casts retained.
Call dtypes and constexpr values are supplied outside both quotations. -/
namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel
open VeriTile.Triton

inductive IndexType where | int32
  deriving DecidableEq

structure Parameters where
  (weight ids output : RegionName)
  (first last : Int)
  (weightStride outputStride n hidden width : Nat)

def callSignature : CallSignature where
  pointers := [("weight",.fp32),("input_ids",.i32),("out",.fp32)]
  scalars := [("vob_start_id",.i64), ("vob_end_id",.i64), ("stride_weight_seq",.i64), ("stride_out_seq",.i64), ("n_ctx",.i64)]
  constexpr := ["hiden_size", "BLOCK_DMODEL", "BLOCK_N", "BLOCK_NN"]
  readPorts := ["weight", "input_ids"]
  writePorts := ["out"]

/-- Unchanged Python `embedding_kernel_repaired`, including every signed64 conversion. -/
def source : IndexType → Parameters → ComputeKernel
  | .int32, p =>
    let weight := p.weight
    let input_ids := p.ids
    let out := p.output
    let hiden_size := p.hidden
    let BLOCK_DMODEL := p.width
    let BLOCK_N := 64
    let BLOCK_NN := 1
    with_triton_call callSignature call_args [vob_start_id := p.first, vob_end_id := p.last, stride_weight_seq := p.weightStride, stride_out_seq := p.outputStride, n_ctx := p.n] in
    triton {
    start_n = tl.program_id(0).to(tl.int64) * BLOCK_N
    offs_nn = start_n + tl.arange(0, BLOCK_NN).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=0).to(tl.int64)
        id_mask = n_ctx_mask & (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = tl.where(id_mask, token_ids, vob_start_id) - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :],
                       mask=load_mask, other=0.0)
        output_rows = tl.where(n_ctx_mask, offs_seq, 0)
        tl.store(out + output_rows[:, None] * stride_out_seq + offs_d[None, :],
                 vecs, mask=store_mask)
    }

/-- Unchanged Python `embedding_kernel_index_commuted`, including every signed64 conversion. -/
def indexCommutedSource : IndexType → Parameters → ComputeKernel
  | .int32, p =>
    let weight := p.weight
    let input_ids := p.ids
    let out := p.output
    let hiden_size := p.hidden
    let BLOCK_DMODEL := p.width
    let BLOCK_N := 64
    let BLOCK_NN := 1
    with_triton_call callSignature call_args [vob_start_id := p.first, vob_end_id := p.last, stride_weight_seq := p.weightStride, stride_out_seq := p.outputStride, n_ctx := p.n] in
    triton {
    start_n = BLOCK_N * tl.program_id(0).to(tl.int64)
    offs_nn = start_n + tl.arange(0, BLOCK_NN).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=0).to(tl.int64)
        id_mask = n_ctx_mask & (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = tl.where(id_mask, token_ids, vob_start_id) - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :],
                       mask=load_mask, other=0.0)
        output_rows = tl.where(n_ctx_mask, offs_seq, 0)
        tl.store(out + output_rows[:, None] * stride_out_seq + offs_d[None, :],
                 vecs, mask=store_mask)
    }

def grid (n : Nat) : Nat := (n+63)/64

end VeriTile.Bench.Repairs.EmbeddingTritonKernel
