import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPShape
import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPWitness
/-! Complete accepted pre_load schedule, all K blocks and genuine edge masks.
Native observed dotAcc is a separate graph, not a value-preserving fusion axiom. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

specification source_backend_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hExact : p.inner=p.blockInner*numKBlocks)
    (hStride : p.outputColumnStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec (backend rest cfg D (softwareDotAcc cfg)) (source .source p) s=some final ∧
      Post cfg p s final (accumulated cfg D (softwareDotAcc cfg) .source p s) :=
  correctness rest cfg D (softwareDotAcc cfg) .source p s (output_injective p s hStride hWidth)

specification source_sequential_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hExact : p.inner=p.blockInner*numKBlocks)
    (hStride : p.outputColumnStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec (backend rest cfg (softwareDot cfg) (softwareDotAcc cfg)) (source .source p) s=some final ∧
      Post cfg p s final (accumulated cfg (softwareDot cfg) (softwareDotAcc cfg) .source p s) :=
  correctness rest cfg (softwareDot cfg) (softwareDotAcc cfg) .source p s (output_injective p s hStride hWidth)

specification compiled_dotacc_backend_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc)
    (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hExact : p.inner=p.blockInner*numKBlocks)
    (hStride : p.outputColumnStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec (backend rest cfg (softwareDot cfg) F) (source .accumulator p) s=some final ∧
      Post cfg p s final (accumulated cfg (softwareDot cfg) F .accumulator p s) :=
  correctness rest cfg (softwareDot cfg) F .accumulator p s (output_injective p s hStride hWidth)

specification compiled_sequential_fma_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hExact : p.inner=p.blockInner*numKBlocks)
    (hStride : p.outputColumnStride=1) (hWidth : p.blockColumns≤p.outputRowStride) :
    ∃ final, FP.Structural.exec (backend rest cfg (softwareDot cfg) (softwareDotAcc cfg)) (source .accumulator p) s=some final ∧
      Post cfg p s final (accumulated cfg (softwareDot cfg) (softwareDotAcc cfg) .accumulator p s) :=
  correctness rest cfg (softwareDot cfg) (softwareDotAcc cfg) .accumulator p s (output_injective p s hStride hWidth)

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
theorem complete_masked_two_block_bit_success (rest : Algebra Word) (g : Graph) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue))
      (source g witnessParameters) witnessInitial=some final ∧
      final.mem "C" 12=.mk .fp32 (FP.Value.ofBits .fp32 0x41a00000) ∧
      final.mem "C" 13=witnessInitial.mem "C" 13 ∧
      final.mem "C" 17=witnessInitial.mem "C" 17 ∧
      final.mem "X" 12=witnessInitial.mem "X" 12 := two_block_bit_success rest g
#axiomsClean source_backend_values
#axiomsClean source_sequential_bits
#axiomsClean compiled_dotacc_backend_values
#axiomsClean compiled_sequential_fma_bits
#axiomsClean complete_masked_two_block_bit_success
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPCorrect
