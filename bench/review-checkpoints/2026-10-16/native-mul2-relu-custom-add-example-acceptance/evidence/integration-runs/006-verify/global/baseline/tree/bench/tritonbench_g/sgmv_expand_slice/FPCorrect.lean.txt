import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPShape
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPFusedSoftware

/-! Full accepted K loop, including metadata, modulo gathers, strides, slice
and masked stores. Source dot+add and the separately observed candidate-compiled
dotAcc graph have independent value contracts. No all-bit compiler rewrite or
native MMA/software FMA equality is assumed. -/
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

specification source_backend_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hEven : p.inner=p.blockInner*numKBlocks)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest cfg D) (source fmt out p) s=some final ∧
      Post out p s final (outputValues cfg fmt out p (accumulated cfg D fmt p s)) :=
  correctness rest cfg D fmt out p s hinj

specification source_sequential_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hEven : p.inner=p.blockInner*numKBlocks)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest cfg (softwareDot cfg)) (source fmt out p) s=some final ∧
      Post out p s final (outputValues cfg fmt out p (accumulated cfg (softwareDot cfg) fmt p s)) :=
  correctness rest cfg (softwareDot cfg) fmt out p s hinj

/-- Explicit-accumulator graph observed only in the one-comma candidate's
native compilation. The original Python kernel failed before lowering. -/
specification candidate_dotacc_backend_values [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (F : DotAcc)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hEven : p.inner=p.blockInner*numKBlocks)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (fusedBackend rest cfg F) (fusedSource fmt out p) s=some final ∧
      Post out p s final (outputValues cfg fmt out p (fusedAccumulated F fmt p s)) :=
  fused_correctness rest cfg F fmt out p s hinj

specification candidate_sequential_fma_bits [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (fmt : Storage) (out : Output) (p : Parameters) (s : State Word) (numKBlocks : Nat)
    (_hBK : 0<p.blockInner) (_hEven : p.inner=p.blockInner*numKBlocks)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ final, FP.Structural.exec (fusedBackend rest cfg (softwareDotAcc cfg)) (fusedSource fmt out p) s=some final ∧
      Post out p s final (outputValues cfg fmt out p (fusedAccumulated (softwareDotAcc cfg) fmt p s)) :=
  fused_correctness rest cfg (softwareDotAcc cfg) fmt out p s hinj

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
theorem source_two_block_bit_success (rest : Algebra Word) (fmt : Storage) (out : Output) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue (softwareDot .ieeeValue))
      (source fmt out witnessParameters) (witnessInitial fmt)=some final ∧
      final.mem "Y" 77=outputCell out (resultWord out) ∧
      final.mem "Y" 78=(witnessInitial fmt).mem "Y" 78 ∧
      final.mem "Y" 87=(witnessInitial fmt).mem "Y" 87 ∧
      final.mem "X" 42=(witnessInitial fmt).mem "X" 42 := two_block_bit_success rest fmt out

theorem candidate_two_block_bit_success (rest : Algebra Word) (fmt : Storage) (out : Output) :
    ∃ final, FP.Structural.exec (fusedBackend rest .ieeeValue (softwareDotAcc .ieeeValue))
      (fusedSource fmt out witnessParameters) (witnessInitial fmt)=some final ∧
      final.mem "Y" 77=outputCell out (resultWord out) ∧
      final.mem "Y" 78=(witnessInitial fmt).mem "Y" 78 ∧
      final.mem "Y" 87=(witnessInitial fmt).mem "Y" 87 ∧
      final.mem "X" 42=(witnessInitial fmt).mem "X" 42 := fused_two_block_bit_success rest fmt out

/-- Distinct software graph results, not a claimed GPU counterexample. -/
theorem software_graph_difference :
    FP.add .ieeeValue (counterCarry counterLane)
      ((softwareDot .ieeeValue).apply (some .fp32) .fp16 .fp16 .fp32 {} (batch := [])
        counterLeft counterRight counterLane)=FP.Value.ofBits .fp32 0x4b800000 ∧
    (softwareDotAcc .ieeeValue).apply (some .fp32) .fp16 .fp16 .real {} (batch := [])
      counterLeft counterRight counterCarry counterLane=FP.Value.ofBits .fp32 0x4b7fffff :=
  no_unconditional_dot_add_dotacc_law
#axiomsClean source_backend_values
#axiomsClean source_sequential_bits
#axiomsClean candidate_dotacc_backend_values
#axiomsClean candidate_sequential_fma_bits
#axiomsClean source_two_block_bit_success
#axiomsClean candidate_two_block_bit_success
#axiomsClean software_graph_difference
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPCorrect
