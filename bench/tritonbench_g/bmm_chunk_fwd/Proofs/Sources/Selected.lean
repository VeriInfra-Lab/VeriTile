import bench.tritonbench_g.bmm_chunk_fwd.Kernels
namespace VeriTile.Bench.Repairs.BmmChunkFwd.Kernels
open VeriTile.Triton
def mainSource
    (out_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
originalKernel .fp32 .fp32 out_dtype .fp32 a_ptr b_ptr out_ptr seq_idx_ptr seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_b_batch stride_b_seqlen stride_b_head stride_bk stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K IS_CAUSAL HAS_SEQ_IDX
end VeriTile.Bench.Repairs.BmmChunkFwd.Kernels
