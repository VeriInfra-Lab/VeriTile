import bench.repairs.tritonbench_g.matrix_transpose.Kernels
import VeriTile.Meta.StatementAudit

/-! Previous full programs are proof-only preservation witnesses. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.SourcePreservation
open VeriTile.Triton

def previous_source (M Out : RegionName) (m d : Nat) : ComputeKernel := triton {
  tl.region M = tl.float16, Out = tl.float16
  size_m_arange = tl.arange(0, $(m))
  d_head_arange = tl.arange(0, $(d))
  matrix_ptr = M + size_m_arange[:, None] * $(max d 1)
    + d_head_arange[None, :] * $(1)
  out_ptr = Out + d_head_arange[:, None] * $(max m 1)
    + size_m_arange[None, :] * $(1)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}

def previous_indexCommutedSource (M Out : RegionName) (m d : Nat) : ComputeKernel := triton {
  tl.region M = tl.float16, Out = tl.float16
  size_m_arange = tl.arange(0, $(m))
  d_head_arange = tl.arange(0, $(d))
  matrix_ptr = M + $(max d 1) * size_m_arange[:, None]
    + d_head_arange[None, :] * $(1)
  out_ptr = Out + d_head_arange[:, None] * $(max m 1)
    + size_m_arange[None, :] * $(1)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}


theorem source_preserved (M Out : RegionName) (m d : Nat) : source M Out m d = previous_source M Out m d := rfl
theorem indexCommutedSource_preserved (M Out : RegionName) (m d : Nat) : indexCommutedSource M Out m d = previous_indexCommutedSource M Out m d := rfl

#axiomsClean source_preserved
#axiomsClean indexCommutedSource_preserved
end VeriTile.Bench.Repairs.MatrixTranspose.SourcePreservation
