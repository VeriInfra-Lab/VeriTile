import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Support.Source

namespace VeriTile.Bench.Repairs.BmmChunkFwd
open Lean VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
namespace Kernels

/- Complete admitted FP32 main source. The two-argument dot and following
outer accumulator addition retain the original source graph. -/
/- The complete selected source is shared through Sources.Selected. -/

/-- Existing integer head-index candidate; all floating operations, causal
control, sequence filtering, output conversion and masks are unchanged. -/
def mainIndexCommuted
    (out_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
    (seq_idx_ptr : Region .int)
    (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      stride_seq_idx_batch stride_seq_idx_seqlen
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) : ComputeKernel :=
indexCommutedKernel .fp32 .fp32 out_dtype .fp32 a_ptr b_ptr out_ptr seq_idx_ptr seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak stride_b_batch stride_b_seqlen stride_b_head stride_bk stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn stride_seq_idx_batch stride_seq_idx_seqlen BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K IS_CAUSAL HAS_SEQ_IDX

def bindMain (version : Bool) (h : Host.Call) (c : Host.Config) :=
  (if version then mainIndexCommuted else mainSource)
    (Host.outputDType h.aStorage h.requestedOutput) h.A h.B h.Out h.Seq
    h.shape.seqlen h.shape.chunk h.shape.K h.shape.groups
    (Host.prepared h.shape h.aStrides).batch (Host.prepared h.shape h.aStrides).row
    (Host.prepared h.shape h.aStrides).head (Host.prepared h.shape h.aStrides).column
    (Host.prepared h.shape h.bStrides).batch (Host.prepared h.shape h.bStrides).row
    (Host.prepared h.shape h.bStrides).head (Host.prepared h.shape h.bStrides).column
    ((h.shape.chunks * h.shape.groups) * h.shape.chunk * h.shape.chunk)
    (h.shape.groups * h.shape.chunk * h.shape.chunk)
    (if h.shape.grouped then h.shape.chunk * h.shape.chunk else 0) h.shape.chunk 1
    (if h.hasSeq then h.seqBatch else 0) (if h.hasSeq then h.seqRow else 0)
    c.M c.N c.K h.causal h.hasSeq


def fp32Source := bindMain Bool.false
def indexCommutedKernel := bindMain Bool.true

/-- Existing general typed source retained for the accepted proof library. -/
abbrev source (h : Host.Call) (c : Host.Config) := h.kernel c Bool.false
abbrev configurations := Host.configurations

theorem fp32Source_eq (h : Host.Call) (c : Host.Config)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    fp32Source h c=source h c := by
  simp only [fp32Source,bindMain,source,Host.Call.kernel,ha,hb,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,mainSource,
    Bool.false_eq_true,↓reduceIte] <;> rfl

/-- The actual final 32 by 64 cleanup source. Int64 casts have the same
mathematical erasure as main; NativeCleanup proves finite source arithmetic. -/
def clearUpperKernel (Out : RegionName) (chunk_size chunk_heads : Nat) : ComputeKernel :=
  VeriTile.Bench.Repairs.BmmChunkFwd.clearUpperKernel .fp32 Out chunk_size chunk_heads 32 64

#axiomsClean fp32Source_eq
end Kernels
end VeriTile.Bench.Repairs.BmmChunkFwd
