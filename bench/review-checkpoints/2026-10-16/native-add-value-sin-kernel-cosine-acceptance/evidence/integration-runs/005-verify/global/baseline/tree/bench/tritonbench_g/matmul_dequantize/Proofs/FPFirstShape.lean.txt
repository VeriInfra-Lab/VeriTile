import bench.tritonbench_g.matmul_dequantize.Proofs.FPFirstComplete
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst
open VeriTile.Triton
noncomputable section
/-- Signed packed input is shifted before its low32-bit mask; the K-loop
body does not cast negative packed words to Nat before the shift. -/
theorem source_signed_shape (p : Parameters) :
    (body p)[3]? = some (.assign .int [p.blockInner,p.blockColumns] "b"
      (.alg (Op.intBitAnd 32 .scalarR
        (Op.intShiftRight Broadcast.nil.consR.consSame
          (.ref .int [p.blockInner,p.blockColumns] "b")
          (.expandDim ⟨1, by change 1 < 2; decide⟩ (.ref .nat [p.blockInner] "shifter"))) (.constInt 15)))) := by rfl
#axiomsClean source_signed_shape
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst
