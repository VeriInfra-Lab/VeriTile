import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.int8_quantization.IO
import bench.tritonbench_g.int8_quantization.Proofs.RealInterface

namespace VeriTile.Bench.Repairs.Int8Quantization.LogicalReader
open VeriTile.Triton FP.Structural Revision002
open scoped ProgramIO
noncomputable section
set_option autoImplicit false

/-- Full source schedule, FP32 input/scale and integer output, exact masks,
windows and empty scratch are visible in IO.lean. -/
def quantizationIO (rest : Algebra ℝ) (p : Parameters) :=
  QuantizationIO.quantizationIO (Math.algebra rest) Math.primitives p

/-- Active logical lanes determine the maximum; tails never contribute. -/
def blockMaximum (p : Parameters) (values : QuantizationIO.Inputs ℝ p)
    (group : Fin (QuantizationIO.count p)) : ℝ :=
  RealSpec.maxAbs ((((List.finRange p.width).filter
    (fun j => decide (QuantizationIO.active p group.val j.val))).map
      (QuantizationIO.raw p values group)).map
        (fun x => RealSpec.factor p.query p.channels*x))

/-- Both output channels: nearest-away integer codes and max/127 scales.
The original total exceptional-block branch is retained. -/
def formula (p : Parameters) (values : QuantizationIO.Inputs ℝ p) : QuantizationIO.Outputs ℝ p := by
  classical
  exact fun o => match o with
  | ⟨0,_⟩ => fun i => if QuantizationIO.FiniteBlock p values i.1 then
      RealSpec.nearestAway (127*(RealSpec.factor p.query p.channels*
        QuantizationIO.raw p values i.1 i.2.1)/blockMaximum p values i.1)
    else (0 : Int)
  | ⟨1,_⟩ => fun i => if QuantizationIO.FiniteBlock p values i.1 then
      blockMaximum p values i.1/127 else (0 : ℝ)


/-- Independent total max-absolute quantization formula for every block;
invalid blocks yield zero. The complete launch, source access safety and exact
exterior frame are conclusions of the shared IO judgment. -/
theorem per_block_int8_real (p : Parameters) (positive : 0 < p.width)
    (block : 0 < p.blockRows) (factor : p.factor = RealSpec.factor p.query p.channels)
    (rest : Algebra ℝ) : Spec.Real (quantizationIO rest p ⊨ formula p) :=
  QuantizationIO.real_implements p positive block factor rest

end
end VeriTile.Bench.Repairs.Int8Quantization.LogicalReader

namespace VeriTile.Bench.Repairs.Int8Quantization.LogicalReader
open VeriTile.Triton FP.Structural Revision002
open scoped LogicalProgramIO
noncomputable section

def logicalMaximum (p : Parameters) (xs : Fin (QuantizationIO.count p) → Fin p.width → ℝ) (group : Fin (QuantizationIO.count p)) : ℝ :=
  RealSpec.maxAbs ((((List.finRange p.width).filter (fun j => decide (QuantizationIO.active p group.val j.val))).map
    (xs group)).map (fun x => RealSpec.factor p.query p.channels*x))
def logicalFormula (p : Parameters) (xs : Fin (QuantizationIO.count p) → Fin p.width → ℝ) : QuantizationIO.LogicalOutputs ℝ p := by
  classical
  exact {
    codes := fun group lane => if QuantizationIO.logicalFiniteBlock p xs group then
      RealSpec.nearestAway (127*(RealSpec.factor p.query p.channels*xs group lane)/logicalMaximum p xs group) else 0
    scales := fun group => if QuantizationIO.logicalFiniteBlock p xs group then logicalMaximum p xs group/127 else 0 }
theorem full_judgment_iff (p : Parameters) (rest : Algebra ℝ) :
    (QuantizationIO.logicalIO (Math.algebra rest) Math.primitives p ⊨ logicalFormula p) ↔ (quantizationIO rest p).Implements (formula p) := by
  have target : (QuantizationIO.logicalIO (Math.algebra rest) Math.primitives p).lift (logicalFormula p) = formula p := by
    funext xs o; fin_cases o <;> funext j <;> rfl
  change (quantizationIO rest p).Implements _ ↔ _
  rw [target]

end
end VeriTile.Bench.Repairs.Int8Quantization.LogicalReader
