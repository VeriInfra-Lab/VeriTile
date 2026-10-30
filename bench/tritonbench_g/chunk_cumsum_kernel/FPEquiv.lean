import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

/-- ADD-COMMUTE applies only to independently computed carry/chunk-sum pairs
that feed a later output. `CarryDomain.iff_inputs` proves this is exactly the
previous reached-site domain. Final carry and unspecified padding stay free.
The fixed strict FP32 theory supplies the rule; the empirical rule set is empty. -/
specification memory_equiv {α : Type} [Inhabited α]
    (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (model : FP.Guarded.Models [] ops.algebra D)
    (input output : RegionName) (T : Nat) (cfg : Config) (pid : Nat) :
    original ops D input output T cfg pid ≡[[]] sumLeft ops D input output T cfg pid :=
  ProgramPair.memory_equiv ops D model input output T cfg pid

#print_fp_assumptions memory_equiv
#specNonCircular CarryDomain.Finite avoiding [repairedKernel,sumLeftKernel,Rewrite.FiniteCarry,FP.Structural.run]
#axiomsClean memory_equiv
#stmtSurfaceSubset memory_equiv ⊆
  [FPRow.Ops,FP.Guarded.Domain,FP.Guarded.Models,Spec.RuleEntry,FP.GuardedFragment,
   FPRow.Ops.algebra,RegionName,Config,
   original,sumLeft,Spec.Equiv,Spec.FloatingPoint,
   originalIO,ProgramIO.Implementation,ProgramIO.Implementation.programSyntax,
   ProgramIO.InputBindingMode.decoded,Spec.ProgramSyntax.Statement,PublicFP.domain]
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.FPEquiv
