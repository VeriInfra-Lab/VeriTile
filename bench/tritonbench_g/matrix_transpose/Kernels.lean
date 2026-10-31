import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure

/-! The repaired FP16 Python source and current integer-index candidate.
All four source strides are explicit; the host defaults are contiguous. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose
open VeriTile.Triton

def source (M Out : RegionName) (m d : Nat)
    (matrix_stridex : Nat := max d 1) (matrix_stridey : Nat := 1)
    (out_stridex : Nat := max m 1) (out_stridey : Nat := 1) : ComputeKernel := triton {
  tl.ports inputs=[M], outputs=[Out]
  tl.region M = tl.float16, Out = tl.float16
  size_m_arange = tl.arange(0, $(m))
  d_head_arange = tl.arange(0, $(d))
  matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
    + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
    + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}

def indexCommutedSource (M Out : RegionName) (m d : Nat)
    (matrix_stridex : Nat := max d 1) (matrix_stridey : Nat := 1)
    (out_stridex : Nat := max m 1) (out_stridey : Nat := 1) : ComputeKernel := triton {
  tl.ports inputs=[M], outputs=[Out]
  tl.region M = tl.float16, Out = tl.float16
  size_m_arange = tl.arange(0, $(m))
  d_head_arange = tl.arange(0, $(d))
  matrix_ptr = M + $(matrix_stridex) * size_m_arange[:, None]
    + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
    + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}

end VeriTile.Bench.Repairs.MatrixTranspose
