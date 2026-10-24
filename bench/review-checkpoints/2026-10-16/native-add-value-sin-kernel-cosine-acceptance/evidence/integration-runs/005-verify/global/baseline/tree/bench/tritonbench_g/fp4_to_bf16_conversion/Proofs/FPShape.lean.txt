import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.Fp4ToBf16

def rawDSL (p : Parameters) := scaledCandidate p.input p.scales p.output p.count p.block p.groups

/-- Whole-kernel parser equality, including native-width signed16 casts. -/
theorem source_exact (p : Parameters) : source p = rawDSL p := rfl

theorem raw_dsl_signed16_exact (p : Parameters) :
    ((rawDSL p).surfaceBody.drop 32 |>.take 1) = offsetCode p := rfl

/-- The old local replacement is now redundant: the complete current parser
result already has exactly that AST. This is an equality, not an evaluator
assumption; the old frozen source/compiler receipt is archived separately. -/
theorem previous_adapted_template_eq (p : Parameters) :
    source p = (match rawDSL p with
      | .mk inputs outputs body => .mk inputs outputs
          (body.take 32 ++ offsetCode p ++ body.drop 33)) := rfl

theorem full_source_length (p : Parameters) :
    (source p).surfaceBody.length = 47 := rfl

theorem group_shapes (p : Parameters) : p.block/16 = p.groups ∧ p.block*2/32 = p.groups := by
  constructor
  · simp [Parameters.block]
  · simp [Parameters.block, Nat.mul_assoc]

theorem active_byte_i16_exact : ∀ b : BitVec 8,
    signed16 (signed16 (b.toNat : Int)-127) = (b.toNat : Int)-127 := by decide +kernel

theorem nan_materialization : (Bits.narrow ⟨0x7fc00000⟩).bits = 0x7fc0 := by decide +kernel

#axiomsClean source_exact
#axiomsClean raw_dsl_signed16_exact
#axiomsClean previous_adapted_template_eq
#axiomsClean full_source_length
#axiomsClean group_shapes
#axiomsClean active_byte_i16_exact
#axiomsClean nan_materialization
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
