import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOSafety
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOResult
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IORealResult

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOComplete
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section

theorem native_implements {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (p : Parameters) (pid : Nat) :
    (io rest P p pid).ImplementsOn (Interface.SeedDomain rest P p pid)
      (nativeOutputs rest P p pid) :=
  UndefProgramIO.ImplementsOn.intro (IOSafety.whole rest P p pid) (IOResult.outcome rest P p pid)

theorem real_satisfies (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    (io rest Real.primitives p pid).SatisfiesOn
      (fun vals=>Interface.SeedDomain rest Real.primitives p pid vals ∧ ProbabilityDomain p pid vals.memory)
      (fun vals outputs=>categorical p pid vals.memory (outputs (0 : Fin 1))) := by
  apply UndefProgramIO.SatisfiesOn.intro
  · intro vals domain
    exact IOSafety.whole rest Real.primitives p pid vals domain.1
  · intro vals domain s bound
    obtain ⟨t,ran,outputs,frame⟩ := IOResult.outcome rest Real.primitives p pid vals domain.1 s bound
    refine ⟨t,_,ran,outputs,?_,frame⟩
    exact IORealResult.categorical_result rest p pid vals ((job rest Real.primitives p pid).initial s)
      (IOFacts.entry_binding rest Real.primitives p pid vals s bound) rfl domain.2

#axiomsClean native_implements
#axiomsClean real_satisfies
end
end VeriTile.Bench.Repairs.MultinomialSampling.IOComplete
