import bench.tritonbench_g.matmul_dequant_int4.Proofs.FPTyped
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
/-- The source operand tree keeps the signed load registers through both
arithmetic shifts and fixed-width masks; conversion happens after subtraction. -/
theorem source_signed_shape (p : Parameters) :
    (source p).surfaceBody[16]? = some (.assign .fp16 [p.blockRows, p.blockColumns]
      "fp_weight" (.alg (.mul .fp16 Broadcast.nil.consSame.consSame
        (.withPrecision .fp16 .fp32 (.castFloat .real .fp16 (.intToReal (.sub .int Broadcast.nil.consSame.consSame
          (Op.intBitAnd 32 .scalarR
            (Op.intShiftRight Broadcast.nil.consR.consSame
              (.ref .int [p.blockRows, p.blockColumns] "int32_b")
              (.ref .nat [p.blockRows, 1] "b_shift")) (.constInt 15))
          (Op.intBitAnd 32 .scalarR
            (Op.intShiftRight Broadcast.nil.consSame.consR
              (.ref .int [p.blockRows, p.blockColumns] "zp_b")
              (.ref .nat [1, p.blockColumns] "bzp_shift")) (.constInt 15))))))
        (.withPrecision .fp16 .fp32 (.ref .fp16 [p.blockRows, p.blockColumns] "scale_b"))))) := by rfl
#axiomsClean source_signed_shape
end
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
