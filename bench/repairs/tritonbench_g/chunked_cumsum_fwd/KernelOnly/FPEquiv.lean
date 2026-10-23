import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.Proofs.PairCorrect

/-! Complete original/scale-first kernel pair on the retained full grid.
Runtime clamp words, both branches, uint64 source indices, both output arrays,
chunk padding, arbitrary inclusive prefix topology and each explicit fused
scan node are preserved. The transformed multiply and FMA ports use the
concrete software value model; no split-to-fused equality is assumed. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.KernelOnly.FPEquiv
open VeriTile.Triton
open Kernel Kernel.Scalar Kernel.Math PairIO
open scoped VeriTile.Spec
noncomputable section

specification chunked_cumsum_scaleFirst_equiv (cfg : FP.Config)
    (M : NativeModel (FP.Value .fp32)) (c : Geometry) (sizes : Sizes c)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : FP.Value .fp32) :
    original cfg M c plan hasBias useSoftplus lower upper ≡[[]]
      scaleFirst cfg M c plan hasBias useSoftplus lower upper :=
  FP.Structural.ProgramIO.Implementation.of_equivalent
    (scaleFirstIO cfg M c plan hasBias useSoftplus lower upper)
    (PairCorrect.equivalent cfg M c sizes plan hasBias useSoftplus lower upper)

#axiomsClean chunked_cumsum_scaleFirst_equiv
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.KernelOnly.FPEquiv
