import bench.tritonbench_g.matrix_transpose.Kernels

/-! Complete selected FP16 sources. The historical untyped kernel and generic
stride-family aliases remain for compatibility. No floating operation or cast
is present; the existing candidate only commutes an integer index product. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose
open VeriTile.Triton

abbrev kernel := VeriTile.Bench.TritonBenchG.MatrixTranspose.kernel
abbrev typedKernel := VeriTile.Bench.TritonBenchG.MatrixTranspose.typedKernel

def source (M Out : RegionName) (m d : Nat) : ComputeKernel := triton {
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

def indexCommutedSource (M Out : RegionName) (m d : Nat) : ComputeKernel := triton {
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

end VeriTile.Bench.Repairs.MatrixTranspose
