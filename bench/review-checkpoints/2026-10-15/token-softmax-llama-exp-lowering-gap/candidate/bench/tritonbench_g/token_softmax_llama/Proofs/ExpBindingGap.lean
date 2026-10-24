import bench.tritonbench_g.token_softmax_llama.Proofs.ActualSource
import bench.tritonbench_g.token_softmax_bloom.Proofs.HalfExpGapReachable

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.ExpLowering
open VeriTile.Triton FP.Structural
open TokenSoftmaxBloom.ExpLoweringGap
noncomputable section

/-- Llama's old complete value formula is already the shared Bloom formula;
there is no new callback or numerical assumption in the two reuse statements.
The normal-fp32 case matches Llama's original Python input format. -/
theorem fp32_source_exponent_inputs (rest : Algebra Word) (exp32 : Word → Word) :
    sourceExpArgument rest exp32 inputA = inputA ∧
    sourceExpArgument rest exp32 inputB = inputB :=
  source_reaches_both_inputs rest exp32

/-- The generic-storage public domain also includes the real-half construction,
with explicitly proved widening and fp32 subtraction, not half-as-fp32 values. -/
theorem fp16_source_exponent_inputs (rest : Algebra Word) (exp32 : Word → Word) :
    halfSourceExpArgument rest exp32 halfA = inputA ∧
    halfSourceExpArgument rest exp32 halfB = inputB :=
  half_source_reaches_both_inputs rest exp32

/-- Refutation only of a universal internal exp-stage binding. It says nothing
about final stored softmax values or actual GPU observations. The complete
source bridge and Python AST audit identify the same original row computation. -/
theorem no_direct_RN_exp_binding_on_fp32_rows (rest : Algebra Word) (exp32 sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue (sourceExpArgument rest exp32 inputA) log2e) = roundedExpA ∧
       sfu (FP.mul .ieeeValue (sourceExpArgument rest exp32 inputB) log2e) = roundedExpB) :=
  reachable_binding_impossible rest exp32 sfu

theorem no_direct_RN_exp_binding_on_half_rows (rest : Algebra Word) (exp32 sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue (halfSourceExpArgument rest exp32 halfA) log2e) = roundedExpA ∧
       sfu (FP.mul .ieeeValue (halfSourceExpArgument rest exp32 halfB) log2e) = roundedExpB) :=
  half_reachable_binding_impossible rest exp32 sfu

#axiomsClean fp32_source_exponent_inputs
#axiomsClean fp16_source_exponent_inputs
#axiomsClean no_direct_RN_exp_binding_on_fp32_rows
#axiomsClean no_direct_RN_exp_binding_on_half_rows
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.ExpLowering
