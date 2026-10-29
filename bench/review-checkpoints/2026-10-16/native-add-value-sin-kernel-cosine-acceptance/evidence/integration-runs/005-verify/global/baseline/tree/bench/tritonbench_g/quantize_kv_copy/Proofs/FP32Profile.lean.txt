import bench.tritonbench_g.matrix_reduction.Proofs.MachineMax
import VeriTile.Triton.Float.IntegerConversionTotal

/-! Minimal shared fp32 scalar/maximum/integer-conversion profile. This module
contains no benchmark mathematical main or source execution proof. -/
namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
abbrev Word := FP.Value .fp32
/-- Selected fp32 bits, with exact encoded ordering and least-index ties.
Abs produces positive zero for both zero signs, so finite equal maxima have
identical encodings. The source's nonempty reduction check is retained. -/
def maximum {n : Nat} (x : Fin n → Word) : Word :=
  ⟨MatrixReduction.MachineMax.rowMaximum .fp32 (fun i => (x i).bits)⟩

def reduction {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → Values Word .real shape →
      Values Word .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false, x => fun out => maximum (fun i => x (TileShape.insertAxisIndex shape ax out i))
  | Bool.true, x => fun out => maximum (fun i => x (TileShape.replaceAxisIndex shape ax out i))

/-- Unsupported primitives remain in `rest`; execution below proves they are
not used. The undefined signed-conversion policy is universally quantified. -/
noncomputable def backend (rest : Algebra Word) (cfg : FP.Config)
    (undefinedValue : Nat → Word → Int) : Algebra Word :=
  fp32TotalBackend
    { FP.Software.scalarFP32 rest cfg with
      literal := fun p d x => if FP.Software.fp32Context p d && decide (x = 127) then
          FP.Value.ofBits .fp32 0x42fe0000 else (FP.Software.scalarFP32 rest cfg).literal p d x
      reduceMax := fun _ => reduction }
    undefinedValue (fun _ _ => 0)

end VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize
