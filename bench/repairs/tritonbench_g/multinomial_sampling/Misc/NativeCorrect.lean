import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.FPCompatibility
import VeriTile.Meta.Specification

/-! One source implementation, observed through its independent native
recurrence. No alternative kernel or native-to-Real equality is introduced.
Primitives retain casts, canonical Philox10, complete scan/max configurations,
and arbitrary primitive models. `software .fp32 plans` instantiates this same
statement for every existing scan/max plan. No numerical rewrite law is used. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section

theorem multinomial_native_observation {α : Type} [Inhabited α]
    (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) :
    (io rest P p pid).ImplementsOn (Interface.SeedDomain rest P p pid)
      (nativeOutputs rest P p pid) :=
  IOComplete.native_implements rest P p pid

#specNonCircular nativeReference avoiding
  [kernel,source,FP.Structural.exec,FP.Structural.run,SourceValue.rounds,SourceValue.written]
#stmtSurfaceSubset multinomial_native_observation ⊆
  [FP.Structural.Algebra,Primitives,Parameters,io,UndefProgramIO.ImplementsOn,UndefProgramIO.Inputs,UndefProgramIO.Outputs,
   Interface.SeedDomain,nativeOutputs]
#axiomsClean multinomial_native_observation
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling
