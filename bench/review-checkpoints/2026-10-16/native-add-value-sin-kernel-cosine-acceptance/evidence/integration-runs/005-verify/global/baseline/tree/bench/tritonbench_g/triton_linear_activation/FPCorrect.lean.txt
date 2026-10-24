import bench.tritonbench_g.triton_linear_activation.Proofs.FPShape
import bench.tritonbench_g.triton_linear_activation.Proofs.FPWitness
/-! Full accepted grouped exact-K loop, both flags, every activation string,
physical storage and saved preactivation. Source and observed compiled graphs
are separate; the latter is not asserted equivalent to the source on all bits. -/
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

specification source_backend_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (tanh32 erf32 : Word → Word) (divide : Word → Word → Word)
    (fmt saved : Format) (p : Parameters) (s : State Word)
    (hFitM : blockRow p s*p.blockRows+p.blockRows≤p.rows)
    (hFitN : blockCol p s*p.blockColumns+p.blockColumns≤p.columns)
    (hStride : p.outputColStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec
      (backend rest cfg D (softwareDotAcc cfg) tanh32 erf32 divide) (source .source fmt saved p) s=some final ∧
      Post cfg .source tanh32 erf32 divide fmt saved p s final
        (accumulated cfg D (softwareDotAcc cfg) .source fmt p s) :=
  correctness rest cfg D (softwareDotAcc cfg) tanh32 erf32 divide .source fmt saved p s
    (output_injective_of_fit p s hFitM hFitN hStride hWidth)

/-- Explicit dot accumulator and scalar operand order observed in offline TTIR.
For fast_gelu the native witness is the separately recorded constexpr candidate. -/
specification compiled_graph_backend_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (F : DotAcc) (tanh32 erf32 : Word → Word) (divide : Word → Word → Word)
    (fmt saved : Format) (p : Parameters) (s : State Word)
    (hFitM : blockRow p s*p.blockRows+p.blockRows≤p.rows)
    (hFitN : blockCol p s*p.blockColumns+p.blockColumns≤p.columns)
    (hStride : p.outputColStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec
      (backend rest cfg (softwareDot cfg) F tanh32 erf32 divide) (source .accumulator fmt saved p) s=some final ∧
      Post cfg .accumulator tanh32 erf32 divide fmt saved p s final
        (accumulated cfg (softwareDot cfg) F .accumulator fmt p s) :=
  correctness rest cfg (softwareDot cfg) F tanh32 erf32 divide .accumulator fmt saved p s
    (output_injective_of_fit p s hFitM hFitN hStride hWidth)

/-- Additional closed software profile, explicitly restricted to ReLU/identity.
The two backend specifications above retain all original activation strings. -/
specification scalar_activation_sequential_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (g : Graph) (fmt saved : Format) (p : Parameters) (s : State Word)
    (_hActivation : ScalarOnly p.activation)
    (hFitM : blockRow p s*p.blockRows+p.blockRows≤p.rows)
    (hFitN : blockCol p s*p.blockColumns+p.blockColumns≤p.columns)
    (hStride : p.outputColStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec
      (backend rest cfg (softwareDot cfg) (softwareDotAcc cfg) zeroUnary zeroUnary (FP.div cfg))
      (source g fmt saved p) s=some final ∧
      Post cfg g zeroUnary zeroUnary (FP.div cfg) fmt saved p s final
        (accumulated cfg (softwareDot cfg) (softwareDotAcc cfg) g fmt p s) :=
  correctness rest cfg (softwareDot cfg) (softwareDotAcc cfg) zeroUnary zeroUnary (FP.div cfg)
    g fmt saved p s (output_injective_of_fit p s hFitM hFitN hStride hWidth)

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
theorem full_two_block_bit_success (rest : Algebra Word) (g : Graph) (fmt saved : Format) (relu : Bool) :
    ∃ final, FP.Structural.exec
      (backend rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue)
        zeroUnary zeroUnary (FP.div .ieeeValue))
      (source g fmt saved (witnessParameters (if relu then "relu" else "identity")))
      (witnessInitial fmt)=some final ∧
      final.mem "C" 14=outputCell fmt (resultWord fmt) ∧
      final.mem "Saved" 16=outputCell saved (resultWord saved) ∧
      final.mem "C" 0=(witnessInitial fmt).mem "C" 0 ∧
      final.mem "Saved" 0=(witnessInitial fmt).mem "Saved" 0 ∧
      final.mem "X" 12=(witnessInitial fmt).mem "X" 12 := two_block_bit_success rest g fmt saved relu

theorem every_activation_bit_success (rest : Algebra Word) (g : Graph) (a : Activation) :
    ∃ final, FP.Structural.exec
      (backend rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue)
        zeroUnary zeroUnary (FP.div .ieeeValue))
      (source g .half .half (witnessParameters a.name)) (witnessInitial .half Bool.true)=some final ∧
      final.mem "C" 14=outputCell .half (FP.Value.zero .fp32) ∧
      final.mem "X" 12=(witnessInitial .half Bool.true).mem "X" 12 := all_activation_bit_success rest g a
#axiomsClean source_backend_values
#axiomsClean compiled_graph_backend_values
#axiomsClean scalar_activation_sequential_bits
#axiomsClean full_two_block_bit_success
#axiomsClean every_activation_bit_success
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPCorrect
