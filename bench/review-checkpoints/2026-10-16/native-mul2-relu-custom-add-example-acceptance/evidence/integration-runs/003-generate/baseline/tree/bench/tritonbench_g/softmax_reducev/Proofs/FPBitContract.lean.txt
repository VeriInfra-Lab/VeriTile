import bench.tritonbench_g.softmax_reducev.Proofs.FPContract
import bench.tritonbench_g.softmax_reducev.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section

/-- Full-loop binary32-value contract. The exponential function and exact
sum/max tree plans are explicit parameters, not hidden universal numerical
laws. No non-NaN, positive-denominator or output-value assumption is used. -/
theorem software_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (exp32 : Word → Word) (sumPlans maxPlans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) (hN : 0<p.tokens)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (softwareBackend rest cfg exp32 sumPlans maxPlans) (source p) s = some final ∧
      Post p s final (wordOutput cfg exp32 sumPlans maxPlans p s) := by
  simpa only [output_software] using
    correctness (softwareBackend rest cfg exp32 sumPlans maxPlans) p s hN hinj
#axiomsClean software_correctness
end
end VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
