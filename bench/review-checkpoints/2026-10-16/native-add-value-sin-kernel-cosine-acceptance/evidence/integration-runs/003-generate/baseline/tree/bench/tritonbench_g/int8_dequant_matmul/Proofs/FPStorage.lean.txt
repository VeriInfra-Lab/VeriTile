import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section

/-- Every actual loaded byte is interpreted as signed two's complement. -/
theorem byte_inputs_bridge {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (h : ByteInputs p s) :
    (∀ k, k < p.steps → ∀ i : AIndex p, ∃ w : BitVec 8,
      s.mem p.input (aOffset p s k i) = .mk .int w.toInt ∧ aValue p s k i = w.toInt) ∧
    (∀ k, k < p.steps → ∀ i : BIndex p, ∃ w : BitVec 8,
      s.mem p.weight (bOffset p s k i) = .mk .int w.toInt ∧ bValue p s k i = w.toInt) := by
  constructor
  · intro k hk i
    obtain ⟨w, hw⟩ := h.1 k hk i
    exact ⟨w, hw, by simp [aValue, hw, byte_wrap]⟩
  · intro k hk i
    obtain ⟨w, hw⟩ := h.2 k hk i
    exact ⟨w, hw, by simp [bValue, hw, byte_wrap]⟩

theorem dot_operand_ranges {α : Type} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) :
    FP.IntegerDot.valid (α := α) (aValue p s k) ∧ FP.IntegerDot.valid (α := α) (bValue p s k) := by
  constructor <;> intro i
  all_goals constructor
  all_goals first
    | exact BitVec.le_toInt _
    | exact BitVec.toInt_lt

/-- Width annotations are actual signed integer expressions, including wrap
of the accumulator on every iteration. The full five-statement loop remains. -/
theorem source_signed_shape (p : Parameters) :
    (body p)[0]? = some (.assign .int [p.blockRows,p.blockInner] "a"
      (.alg (.intToSigned 8 (.load .int (.ptr (.ref .ptr [p.blockRows,p.blockInner] "A")) .none)))) ∧
    (body p)[1]? = some (.assign .int [p.blockInner,p.blockColumns] "b"
      (.alg (.intToSigned 8 (.load .int (.ptr (.ref .ptr [p.blockInner,p.blockColumns] "B")) .none)))) ∧
    (body p)[2]? = some (.assign .int [p.blockRows,p.blockColumns] "acc_int"
      (.alg (.intToSigned 32 (.add .int (.consSame (.consSame .nil))
        (.ref .int [p.blockRows,p.blockColumns] "acc_int")
        (Op.dotInt (batch := []) (.ref .int [p.blockRows,p.blockInner] "a")
          (.ref .int [p.blockInner,p.blockColumns] "b")))))) := by
  exact ⟨rfl, rfl, rfl⟩

/-- fp32 scale buffers are loaded through fp32 storage channels, before
register widening. This property concerns the source itself. -/
theorem source_scale_storage (p : Parameters) :
    (prelude p).drop 18 = scales p ++
      [.assign .int [p.blockRows,p.blockColumns] "acc_int"
        (.alg (.full [p.blockRows,p.blockColumns] (.constInt 0)))] := by rfl

#axiomsClean byte_inputs_bridge
#axiomsClean dot_operand_ranges
#axiomsClean source_signed_shape
#axiomsClean source_scale_storage
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
