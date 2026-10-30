import bench.repairs.tritonbench_g.seeded_dropout.Proofs.ProgramPair
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.SeededDropout.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Both sources are safe and terminate, every active output agrees, and both
preserve the entire exterior frame. The comparison port is a structural machine
condition; no RNG, finite-value or floating arithmetic rewrite law is assumed. -/
specification index_commuted_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt (some .fp32) .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed) :
    original M k X O n p seed pid ≡[[]] indexCommuted M k X O n p seed pid :=
  ProgramPair.index_commuted_equiv M lt comparison k X O n p seed pid hn hp hs

#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#stmtSurfaceSubset index_commuted_equiv ⊆
  [FP.Structural.Algebra,FP.Structural.Algebra.compareLt,ComputeDType,ComputeDType.fp32,FloatDType.real,SeedKind,RegionName,
   SeedCarrier,gridCount,SeedABI,original,indexCommuted,Spec.Equiv,Spec.FloatingPoint,
   originalIO,ProgramIO.Implementation,ProgramIO.Implementation.programSyntax,
   ProgramIO.InputBindingMode.decoded,Spec.ProgramSyntax.Statement,ProgramIO.Inputs,Spec.RuleEntry]
end
end VeriTile.Bench.Repairs.SeededDropout.FPEquiv
