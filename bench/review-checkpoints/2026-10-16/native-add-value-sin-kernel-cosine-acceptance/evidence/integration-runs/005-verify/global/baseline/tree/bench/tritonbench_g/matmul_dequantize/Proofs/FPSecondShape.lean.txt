import bench.tritonbench_g.matmul_dequantize.Proofs.FPSecondComplete
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedSecond
open VeriTile.Triton
noncomputable section
/-- Actual signed int32 operands survive the shift and fixed-width mask;
there is no Nat clamp before extracting a nibble. -/
theorem source_signed_shape (p : Parameters) :
    (body p)[8]? = some (.assign .int [p.blockInner,p.blockColumns] "int_b"
      (.alg (Op.intBitAnd 32 .scalarR
        (Op.intShiftRight Broadcast.nil.consR.consSame
          (.ref .int [p.blockInner,p.blockColumns] "b")
          (.ref .nat [p.blockInner,1] "b_shift_bits")) (.constInt 15)))) ∧
    (body p)[9]? = some (.assign .int [p.blockInner,p.blockColumns] "int_bzp"
      (.alg (Op.intBitAnd 32 .scalarR
        (Op.intShiftRight Broadcast.nil.consSame.consR
          (.ref .int [p.blockInner,p.blockColumns] "bzp")
          (.ref .nat [1,p.blockColumns] "bzp_shift_bits")) (.constInt 15)))) := by
  constructor <;> rfl
#axiomsClean source_signed_shape
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedSecond
