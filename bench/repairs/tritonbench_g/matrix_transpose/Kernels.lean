import bench.tritonbench_g.matrix_transpose.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! Complete selected FP16 sources. The historical untyped kernel and generic
stride-family aliases remain for compatibility. No floating operation or cast
is present; the existing candidate only commutes an integer index product. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("M",.fp16),("Out",.fp16)]
  scalars := [("matrix_stridex",.i32),("matrix_stridey",.i32),("out_stridex",.i32),("out_stridey",.i32)]
  constexpr := ["SIZE_M", "D_HEAD"]


abbrev kernel := VeriTile.Bench.TritonBenchG.MatrixTranspose.kernel
abbrev typedKernel := VeriTile.Bench.TritonBenchG.MatrixTranspose.typedKernel

def source (M Out : RegionName) (m d : Nat) : ComputeKernel :=
  let SIZE_M : Nat := m
  let D_HEAD : Nat := d
  let matrixStride : Nat := max d 1
  let outputStride : Nat := max m 1
  with_triton_call callSignature call_args [matrix_stridex := matrixStride, matrix_stridey := 1, out_stridex := outputStride, out_stridey := 1] in
  triton {
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    matrix_ptr = M + size_m_arange[:, None] * matrix_stridex + d_head_arange[None, :] * matrix_stridey
    out_ptr = Out + d_head_arange[:, None] * out_stridex + size_m_arange[None, :] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, tl.trans(matrix))
  }

def indexCommutedSource (M Out : RegionName) (m d : Nat) : ComputeKernel :=
  let SIZE_M : Nat := m
  let D_HEAD : Nat := d
  let matrixStride : Nat := max d 1
  let outputStride : Nat := max m 1
  with_triton_call callSignature call_args [matrix_stridex := matrixStride, matrix_stridey := 1, out_stridex := outputStride, out_stridey := 1] in
  triton {
    """The existing Lean variant, commuting only integer index multiplication."""
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    matrix_ptr = M + matrix_stridex * size_m_arange[:, None] + d_head_arange[None, :] * matrix_stridey
    out_ptr = Out + d_head_arange[:, None] * out_stridex + size_m_arange[None, :] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, tl.trans(matrix))
  }

end VeriTile.Bench.Repairs.MatrixTranspose
