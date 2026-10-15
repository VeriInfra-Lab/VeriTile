import bench.tritonbench_g.token_softmax_llama.FPCorrect
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.ExpLowering
open VeriTile.Triton
def source (dtype : FloatDType)
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Logics = DTYPE, Prob_Out = DTYPE
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  row = (tl.load(Logics + cur_head * $(stride_logic_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_logic_bs),
    mask=col_offsets < cur_batch_seq_len, other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    (softmax_output).to(DTYPE), mask=col_offsets < cur_batch_seq_len)
}

open FP.Structural
open TokenSoftmaxBloom.Typed (Parameters)
/-- A separate parser invocation containing this task's complete original
row body and the already-accepted explicit storage annotations. Equality to
Bloom is justified independently by the exact Python JIT/host AST audit. -/
def kernel (d : FloatDType) (p : Parameters) : ComputeKernel :=
  source d p.input p.start p.length p.output p.inputHeadStride p.inputTokenStride
    p.outputHeadStride p.outputTokenStride p.width

theorem parsed_source_eq_bloom (d : FloatDType) (p : Parameters) :
    kernel d p = TokenSoftmaxBloom.Typed.kernel d p := rfl

theorem parsed_source_erases_to_old_llama (d : FloatDType) (p : Parameters) :
    (kernel d p).eraseDType.surfaceBody =
      (token_softmax_surface p.input p.start p.length p.output p.inputHeadStride
        p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width).eraseDType.surfaceBody := by
  rw [parsed_source_eq_bloom]
  exact Typed.source_shape d p

noncomputable section
/-- This preserves the complete old initial-memory/readback/frame contract and
all its parameter domains. It is a source reuse bridge, not a new closed
software exponential implementation or a native conformance theorem. -/
theorem complete_old_row_contract {α : Type} [Inhabited α]
    (M : Algebra α) (d : FloatDType) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hstride : p.outputTokenStride ≠ 0) :
    ∃ t, FP.Structural.exec M (kernel d p) s = some t ∧
      t.mem = (TokenSoftmaxBloom.Typed.reference M d p s).mem ∧
      (∀ i : TileIndex [p.width], i.1.val < TokenSoftmaxBloom.Typed.count p s →
        t.mem p.output (TokenSoftmaxBloom.Typed.outputOffset p s i) =
          .mk d.toTileDType (ofFloat d (TokenSoftmaxBloom.Typed.outputValues M d p s i))) ∧
      (∀ r o, TokenSoftmaxBloom.Typed.Outside p s r o → t.mem r o = s.mem r o) := by
  rw [parsed_source_eq_bloom]
  exact FPCorrect.token_softmax_typed_correctness M d p s hwidth hstride

#axiomsClean parsed_source_eq_bloom
#axiomsClean parsed_source_erases_to_old_llama
#axiomsClean complete_old_row_contract
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama.ExpLowering
