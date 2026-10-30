import bench.tritonbench_g.int8_quantization.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.Int8Quantization.Specification
open VeriTile.Triton FP.Structural Revision002
open scoped LogicalProgramIO
noncomputable section

/-- Only the active logical lanes contribute to each block maximum. -/
def blockMaximum (p : Parameters) (xs : Fin (QuantizationIO.count p) → Fin p.width → ℝ)
    (group : Fin (QuantizationIO.count p)) : ℝ :=
  RealSpec.maxAbs ((((List.finRange p.width).filter (fun j => decide (QuantizationIO.active p group.val j.val))).map
    (xs group)).map (fun x => RealSpec.factor p.query p.channels*x))

def formula (p : Parameters) (xs : Fin (QuantizationIO.count p) → Fin p.width → ℝ) : QuantizationIO.LogicalOutputs ℝ p := by
  classical
  exact {
    codes := fun group lane => if QuantizationIO.logicalFiniteBlock p xs group then
      RealSpec.nearestAway (127*(RealSpec.factor p.query p.channels*xs group lane)/blockMaximum p xs group) else 0
    scales := fun group => if QuantizationIO.logicalFiniteBlock p xs group then blockMaximum p xs group/127 else 0 }


/-- The total two-output block formula, preserving invalid/zero blocks, all active tails and full source frame. -/
specification per_block_int8_real (p : Parameters) (positive : 0<p.width) (block : 0<p.blockRows)
    (factor : p.factor=RealSpec.factor p.query p.channels) (rest : Algebra ℝ) :
    Spec.Real (QuantizationIO.logicalIO (Math.algebra rest) Math.primitives p ⊨ formula p) :=
  (LogicalReader.full_judgment_iff p rest).mpr (LogicalReader.per_block_int8_real p positive block factor rest)
#axiomsClean per_block_int8_real

end
end VeriTile.Bench.Repairs.Int8Quantization.Specification
