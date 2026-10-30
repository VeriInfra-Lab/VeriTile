import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Kernels

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.KernelInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu

/-- Only source geometry, regions, strides and the constexpr activation. -/
structure Parameters where
  (A B C : RegionName)
  (rows inner columns : Nat)
  (strideAM strideAK strideBK strideBN : Nat)
  activation : String

/-- The legacy host-only fields are fixed internally; no Tensor flags enter
this interface or affect the selected source. -/
def Parameters.request (p : Parameters) : Whole.Request where
  A := p.A; B := p.B; C := p.C
  rows := p.rows; inner := p.inner; innerB := p.inner; columns := p.columns
  strideAM := p.strideAM; strideAK := p.strideAK
  strideBK := p.strideBK; strideBN := p.strideBN
  aContiguous := Bool.true
  activation := p.activation

end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.KernelInterface
