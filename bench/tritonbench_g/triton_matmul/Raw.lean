import bench.tritonbench_g.triton_matmul.Proofs.Whole
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
open VeriTile.Triton FPValue

/-- Only raw source regions, geometry, element storage and strides. -/
structure Geometry where
  (A B C : RegionName)
  storage : Storage
  (rows columns inner strideAM strideAK strideBK strideBN : Nat)

def Geometry.request (g : Geometry) : Request where
  A := g.A; B := g.B; C := g.C
  storage := g.storage; storageB := g.storage
  rows := g.rows; columns := g.columns; inner := g.inner; innerB := g.inner
  strideAM := g.strideAM; strideAK := g.strideAK; strideBK := g.strideBK; strideBN := g.strideBN

end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
