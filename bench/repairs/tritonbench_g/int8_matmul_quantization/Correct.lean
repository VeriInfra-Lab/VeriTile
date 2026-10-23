import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Public
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectGemmReal
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectRow
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectHelperResults
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectComposed

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section

theorem correct (s : Shape) (b : Buffers) (sep : b.Separate) (owner callerDevice : Nat)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory ℝ ℝ) (x : MatrixCell s) :
    (wrapper realOps s b owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      .inr (embedReal b.outPrecision (target s (realInput s b m) x.1 x.2)) := by
  rw [wrapper_word realOps s b sep,real_reference]
theorem quantizer_correct (q : QuantBuffers) (sep : q.Separate) (trials : List (QuantInvocation G K))
    (chosen : QuantInvocation G K) (m : Memory ℝ ℝ) (x : QuantCell G K) :
    quantWrapper realOps q trials chosen m (q.dataAddress x)=
      .inl (quantValueTarget (fun k => read32 realOps.toOps m (q.inputAddress x.1 k)) K x.2) := by
  have h := quantWrapper_at realOps q sep trials chosen m (.inl x)
  change quantWrapper realOps q trials chosen m (q.dataAddress x)=.inl (quantValue realOps q chosen.plan m x) at h
  rw [h,quantValue_real]
theorem composed_correct (s : Shape) (q : QuantBuffers) (qsep : q.Separate) (b : Buffers) (sep : b.Separate)
    (qtrials : List (QuantInvocation s.M s.K)) (qchosen : QuantInvocation s.M s.K)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (m : Memory ℝ ℝ) (x : MatrixCell s) :
    (composed realOps s q b qtrials qchosen owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      .inr (embedReal b.outPrecision (target s (realInput s b (quantMemoryReference realOps q qchosen.plan m)) x.1 x.2)) := by
  rw [composed_reference realOps s q qsep b sep]
  unfold composedReference
  rw [real_reference]
namespace Specification
open Direct
abbrev matmul_int8_wrapper_correct := gemm_real_complete
abbrev quantize_int8_perrow_wrapper_correct := row_real_complete
abbrev quantize_int8_wrapper_correct := helper_real_complete
abbrev matmul_quantize_int8_wrapper_correct := composed_real_complete
#axiomsClean matmul_quantize_int8_wrapper_correct
#axiomsClean matmul_int8_wrapper_correct
#axiomsClean quantize_int8_perrow_wrapper_correct
#axiomsClean quantize_int8_wrapper_correct
end Specification

#specNonCircular target avoiding [wrapper,gridRun,wordReference]
#axiomsClean correct
#axiomsClean quantizer_correct
#axiomsClean composed_correct
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
