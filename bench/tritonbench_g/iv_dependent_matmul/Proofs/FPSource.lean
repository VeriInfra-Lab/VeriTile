import bench.tritonbench_g.iv_dependent_matmul.SourceDefinitions
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
theorem source_split (g : Graph) (p : Parameters) :
    (source g p).surfaceBody = prologue p ++
      [.forRangeDyn "k" (.constNat 0) (stopOp p) (.constNat 1) (body g p)] ++ suffix p := by
  cases g <;> rfl
#axiomsClean source_split
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
