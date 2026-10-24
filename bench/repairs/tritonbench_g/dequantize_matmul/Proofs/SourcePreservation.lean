import bench.repairs.tritonbench_g.dequantize_matmul.Kernels
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.LegacyPastedSurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DequantizeMatmul.SourcePreservation
/-- Entire source family, including every runtime stride/count and both block sizes. -/
theorem original : TypedSource.kernel = LegacyPastedSurface.TypedSource.kernel := rfl

/-- Both complete canonical source/candidate programs equal their retained programs. -/
theorem kernel (v : Bool) (p : KernelSource.Parameters) :
    KernelSource.kernel v p = LegacyPastedSurface.KernelSource.kernel v
      ⟨p.b,p.scale,p.output,p.K,p.N,p.BK,p.BN⟩ := by
  cases v <;> rfl
#axiomsClean original
#axiomsClean kernel
end VeriTile.Bench.Repairs.DequantizeMatmul.SourcePreservation
