import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealProgram
import bench.tritonbench_g.chunk_gated_attention.Proofs.HErasure
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedProgram
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram

def parameters (p : Layout) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) : HSource.Parameters :=
  ⟨p,base,gateK,hasSeed,storeFinal,swapped⟩

def source (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) : ComputeKernel :=
  HSource.kernel (parameters p base gateK hasSeed storeFinal swapped) r dot.dtype gate.dtype history.dtype seed final

/-- Checked projection of the complete repaired source. -/
def body (p : Layout) (r : Regions) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) : List Stmt :=
  (HSource.algorithm (parameters p base gateK hasSeed storeFinal swapped) r).body

theorem source_body (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) :
    (source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.body=
      body p r base gateK hasSeed storeFinal swapped := by
  exact congrArg Kernel.body (HSource.Real.source_connection
    (parameters p base gateK hasSeed storeFinal swapped) r dot.dtype gate.dtype history.dtype seed final)

#axiomsClean source_body
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedProgram
