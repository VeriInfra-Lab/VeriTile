import bench.tritonbench_g.seeded_dropout.Proofs.PublicIO
import bench.tritonbench_g.seeded_dropout.IO

namespace VeriTile.Bench.Repairs.SeededDropout.ProgramPair
open VeriTile.Triton FP.Structural FPEquiv
open scoped VeriTile.Spec
noncomputable section

/-- Retain both safe successful executions, complete active outputs and frames.
The comparison and index/seed ABI premises are unchanged structural bindings. -/
theorem index_commuted_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt (some .fp32) .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed) :
    original M k X O n p seed pid ≡[[]] indexCommuted M k X O n p seed pid :=
  ProgramIO.Implementation.of_equivalent (commutedIO M k X O n p seed pid)
    (KernelPublic.equivalent (FP.ProfiledRewrite.engine M) lt comparison
      k X O n p seed pid hn hp hs)

#axiomsClean index_commuted_equiv
end
end VeriTile.Bench.Repairs.SeededDropout.ProgramPair
