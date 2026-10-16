import bench.tritonbench_g.triton_linear_activation.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
abbrev softwareDot := SgmvExpandSlice.FPValue.softwareDot
abbrev softwareDotAcc := SgmvExpandSlice.FPValue.softwareDotAcc

/-- Every unknown activation string is identity. This configuration predicate
isolates the closed scalar ReLU/identity software profile without restricting
the main full-activation backend contract. -/
def ScalarOnly (name : String) : Prop := name≠"tanh" ∧ name≠"gelu" ∧ name≠"fast_gelu"

theorem scalar_only_independent (cfg : FP.Config) (g : Graph) (name : String) (h : ScalarOnly name)
    (t₁ e₁ t₂ e₂ : Word → Word) (d₁ d₂ : Word → Word → Word) (x : Word) :
    activate cfg g t₁ e₁ d₁ name x=activate cfg g t₂ e₂ d₂ name x := by
  rcases h with ⟨ht,he,hf⟩
  simp [activate,ht,he,hf]

/-- A concrete auxiliary callback for witnesses that reach only signed zero.
Values away from zero deliberately return qNaN; this is not a libdevice model. -/
def zeroUnary (x : Word) : Word := if x.isZero then x else FP.Value.qNaN .fp32

theorem maximum_special_values :
    maximumWord .ieeeValue (FP.Value.zero .fp32) (FP.Value.qNaN .fp32)=FP.Value.zero .fp32 ∧
    maximumWord .ieeeValue (FP.Value.qNaN .fp32) (FP.Value.zero .fp32)=FP.Value.zero .fp32 ∧
    maximumWord .ieeeValue (FP.Value.zero .fp32 Bool.true) (FP.Value.zero .fp32)=FP.Value.zero .fp32 := by
  decide +kernel
#axiomsClean scalar_only_independent
#axiomsClean maximum_special_values
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
