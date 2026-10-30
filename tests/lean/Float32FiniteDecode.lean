import VeriTile.Triton.DSL
import VeriTile.Triton.Float.DecoderCompatibility

/-! Regression for the algorithm-facing constant-bitcast path. -/
namespace VeriTile.Bench.Tests.Float32FiniteDecode
open VeriTile.Triton

-- Both signed zeros and both extreme positive subnormals must elaborate
-- through the actual public DSL, including the Option.get proof guard.
def finiteBitcasts (outReg : RegionName) : ComputeKernel := triton {
  positiveZero := tl.bitcast(0x00000000, tl.float32)
  negativeZero := tl.bitcast(0x80000000, tl.float32)
  smallest := tl.bitcast(0x00000001, tl.float32)
  negativeSmallest := tl.bitcast(0x80000001, tl.float32)
  largest := tl.bitcast(0x007fffff, tl.float32)
  normal := tl.bitcast(0x00800000, tl.float32)
  tl.store($(outReg), positiveZero)
  tl.store($(outReg) + 1, negativeZero)
  tl.store($(outReg) + 2, smallest)
  tl.store($(outReg) + 3, negativeSmallest)
  tl.store($(outReg) + 4, largest)
  tl.store($(outReg) + 5, normal)
}

example (outReg : RegionName) : (finiteBitcasts outReg).toAlgorithm?.isOk = Bool.true := by
  simp [finiteBitcasts, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeOp.constOpToAlgorithm?, ComputeOp.constPayload?, ComputeOp.bitcastPayload,
    ComputeOp.constToAlgorithm?, FP.DecoderCompatibility.positive_zero,
    FP.DecoderCompatibility.negative_zero, FP.DecoderCompatibility.smallest_subnormal,
    FP.DecoderCompatibility.negative_subnormal, FP.DecoderCompatibility.largest_subnormal,
    FP.DecoderCompatibility.smallest_normal, bind, Except.bind, Except.isOk]
  rfl

/-- error: tl.bitcast(..., tl.float32): NaN/Inf fp32 (exponent field 255) is not modeled in AlgorithmCorrect_without_Rounding -/
#guard_msgs in
 def infinityBitcast : ComputeKernel := triton {
   x := tl.bitcast(0x7f800000, tl.float32)
 }

/-- error: tl.bitcast(..., tl.float32): NaN/Inf fp32 (exponent field 255) is not modeled in AlgorithmCorrect_without_Rounding -/
#guard_msgs in
 def nanBitcast : ComputeKernel := triton {
   x := tl.bitcast(0x7fc00001, tl.float32)
 }

end VeriTile.Bench.Tests.Float32FiniteDecode

namespace VeriTile.Bench.Tests.Float32FiniteVectorRegression
open VeriTile.Triton

def vectorBitcastStores (outReg : RegionName) (N : Nat) : ComputeKernel := triton {
  offsets := tl.arange(0, $(N))
  negativeZero := tl.bitcast(0x80000000, tl.float32)
  smallest := tl.bitcast(0x00000001, tl.float32)
  tl.store($(outReg) + offsets, negativeZero)
  tl.store($(outReg) + $(N) + offsets, smallest)
}

theorem vector_store_projection (outReg : RegionName) (N : Nat) :
    (vectorBitcastStores outReg N).toAlgorithm?.isOk = Bool.true := by
  simp [vectorBitcastStores, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeOp.constOpToAlgorithm?, ComputeOp.constPayload?, ComputeOp.bitcastPayload,
    ComputeOp.constToAlgorithm?, FP.DecoderCompatibility.negative_zero,
    FP.DecoderCompatibility.smallest_subnormal, bind, Except.bind, Except.isOk]
  rfl

/-- The actual expansion stores [N]-element vectors and broadcasts both finite
fp32 scalars; this is not merely another scalar-store elaboration check. -/
theorem vector_store_shapes (outReg : RegionName) (N : Nat) :
    (match vectorBitcastStores outReg N with | .mk _ _ body => body).drop 3 =
      [ ComputeStmt.store .real [N]
          (.region outReg (.ref .nat [N] "offsets"))
          (.alg ((Op.ref .real [] "negativeZero").broadcast [N])) .none,
        ComputeStmt.store .real [N]
          (.region outReg (.add .nat .scalarL (.constNat N) (.ref .nat [N] "offsets")))
          (.alg ((Op.ref .real [] "smallest").broadcast [N])) .none ] := by
  rfl

#print axioms vector_store_projection
#print axioms vector_store_shapes
end VeriTile.Bench.Tests.Float32FiniteVectorRegression
