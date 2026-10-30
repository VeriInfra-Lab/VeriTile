import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Kernels
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.LegacyPastedSurface
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.SourcePreservation
open VeriTile.Triton
/-- Exact whole-body identities retain all earlier source events. The explicit
call ports now describe the existing canonical IO rather than the old empty
collector result; no execution or mathematical-domain condition is added. -/
theorem original_body
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) : (source dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).surfaceBody =
    (LegacySource.source dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).surfaceBody := rfl

/-- Complete optional-state equality includes failure for every backend. -/
theorem original
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) {A : Type} [Inhabited A] (M : FP.Structural.Algebra A) (s : FP.Structural.State A) :
    FP.Structural.exec M (source dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK) s =
      FP.Structural.exec M (LegacySource.source dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK) s := by
  rfl

#axiomsClean original_body
#axiomsClean original

theorem candidate_body
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) : (scaleFirst dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).surfaceBody =
    (LegacySource.scaleFirst dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).surfaceBody := rfl

/-- Complete optional-state equality includes failure for every backend. -/
theorem candidate
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) {A : Type} [Inhabited A] (M : FP.Structural.Algebra A) (s : FP.Structural.State A) :
    FP.Structural.exec M (scaleFirst dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK) s =
      FP.Structural.exec M (LegacySource.scaleFirst dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK) s := by
  rfl

#axiomsClean candidate_body
#axiomsClean candidate

end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.SourcePreservation
