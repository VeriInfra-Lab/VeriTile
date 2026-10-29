import bench.tests.FP4Source
import bench.tritonbench_g.fp4_to_bf16.FPCorrect
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton
/-- Task port is the exact pinned upstream-helper-inlining probe. -/
theorem source_matches_frozen_probe (p : Parameters) :
    source p = VeriTile.Tests.FP4Source.source p.input p.output p.count p.block := by rfl
#axiomsClean source_matches_frozen_probe
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
