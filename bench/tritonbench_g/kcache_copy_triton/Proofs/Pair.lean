import bench.tritonbench_g.kcache_copy_triton.Proofs.PublicIO
import bench.tritonbench_g.kcache_copy_triton.IO

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.PairProof
open VeriTile.Triton FP.Structural FPEquiv
open scoped VeriTile.Spec
noncomputable section

/-- Reuse the complete two-source execution/safety/output/frame proof. -/
theorem equivalent {α : Type} [Inhabited α] (M : Algebra α)
    (p : Source.Parameters) (ids : Nat → Nat) :
    original M p ids ≡[[]] candidate M p ids :=
  ProgramIO.Implementation.of_equivalent (commutedIO M p ids)
    (CopyIO.equivalent (FP.ProfiledRewrite.engine M) p ids)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.PairProof
