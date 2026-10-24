import bench.tritonbench_g.softmax_reducev.Proofs.FPShape
import bench.tritonbench_g.softmax_reducev.Proofs.FPWitness

/-! Full accepted per-program online loop, initial-memory value specification,
source success, readback and frame. The software profile closes binary32 scalar
arithmetic and specified reduction trees; exp32 remains explicit. This is not
an all-input native GPU or mathematical-softmax equivalence claim. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPCorrect
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section

specification whole_backend_values {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (hN : 0 < p.tokens)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec M (source p) s = some final ∧
      Post p s final (outputValues M p (finalAccumulator M p s)) :=
  correctness M p s hN hinj

specification fp32_software_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (exp32 : Word → Word) (sumPlans maxPlans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) (hN : 0 < p.tokens)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (softwareBackend rest cfg exp32 sumPlans maxPlans) (source p) s = some final ∧
      Post p s final (wordOutput cfg exp32 sumPlans maxPlans p s) :=
  software_correctness rest cfg exp32 sumPlans maxPlans p s hN hinj

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
theorem two_block_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec
      (softwareBackend rest .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed)
      (source witnessParameters) witnessInitial = some final ∧
      final.mem "O" 14 = .mk .fp32 (FP.Value.ofBits .fp32 0x40800000) ∧
      final.mem "O" 15 = .mk .fp32 (FP.Value.ofBits .fp32 0x40a00000) ∧
      final.mem "O" 16 = witnessInitial.mem "O" 16 ∧
      final.mem "V" 48 = witnessInitial.mem "V" 48 :=
  nonempty_bit_success rest

theorem accepted_two_block_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec
      (softwareBackend rest .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed)
      (source witnessParameters) acceptedWitnessInitial = some final ∧
      final.mem "O" 14 = .mk .fp32 (FP.Value.ofBits .fp32 0x40900000) ∧
      final.mem "O" 15 = .mk .fp32 (FP.Value.ofBits .fp32 0x40b00000) ∧
      final.mem "O" 16 = acceptedWitnessInitial.mem "O" 16 ∧
      final.mem "V" 48 = acceptedWitnessInitial.mem "V" 48 := accepted_bit_success rest
#axiomsClean accepted_two_block_bit_success

#axiomsClean whole_backend_values
#axiomsClean fp32_software_values
#axiomsClean two_block_bit_success
end
end VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPCorrect
