import bench.repairs.tritonbench_g.multinomial_sampling.PairIO

namespace VeriTile.Bench.Repairs.MultinomialSampling.UnrolledPair
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem equivalent (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat) :
    Equivalent (original rest P p pid) (candidate rest P p pid) := by
  intro vals domain cap backed initial bound
  obtain ⟨safe,final,ran,_,frame⟩:=IOComplete.native_implements rest P p pid vals domain initial bound
  have leftSafe : (job rest P p pid).Safe cap initial := by
    simpa only [io,IOUndef.addUndefined,memoryIO,scheduleSafe_singleton] using safe cap backed
  have leftRun : (job rest P p pid).run initial=some final := by
    simpa only [io,IOUndef.addUndefined,memoryIO,runSchedule_singleton] using ran
  have runs : (job rest P p pid).run initial=
      (candidate rest P p pid).invocation.run initial :=
    Unrolled.execution_equal (backend rest P) p _
  refine ⟨leftSafe,?_,final,final,leftRun,runs.symm.trans leftRun,?_,frame,frame⟩
  · exact (Unrolled.safety_equal (backend rest P) p cap _).mp leftSafe
  · intro o j active
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MultinomialSampling.UnrolledPair
