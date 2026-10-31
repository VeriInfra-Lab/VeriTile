import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_Kernels

/-! The source and observed compiled arithmetic graphs remain distinct. The
checked wrapper changes admission only, retaining both activation classes. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu

noncomputable abbrev formula := VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.mathematical

end VeriTile.Bench.Repairs.MatmulLeakyrelu
