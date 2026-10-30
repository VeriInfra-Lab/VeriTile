import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.PublicFP
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.IO

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.ProgramPair
open VeriTile Triton FP.Structural FPEquiv
open scoped VeriTile.Spec
noncomputable section

/-- The shared view retains both successful safe schedules and both frames.
The original strict model and independent finite carry domain remain explicit. -/
theorem memory_equiv {α : Type} [Inhabited α]
    (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (model : FP.Guarded.Models [] ops.algebra D)
    (input output : RegionName) (T : Nat) (cfg : Config) (pid : Nat) :
    original ops D input output T cfg pid ≡[[]] sumLeft ops D input output T cfg pid :=
  ProgramIO.Implementation.of_equivalentOn (sumLeftIO ops input output T cfg pid)
    (PublicFP.equivalent ops D model input output T cfg pid)

#axiomsClean memory_equiv
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.ProgramPair
