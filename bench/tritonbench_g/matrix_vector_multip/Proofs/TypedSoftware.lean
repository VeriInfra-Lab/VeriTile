import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedIO
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Explicit ascending-index software tree. Native tree binding is separate. -/
def sequentialSum (cfg : FP.Config) {shape : TileShape} (ax : Fin shape.length)
    (keepDims : Bool) (x : Values (FP.Value .fp32) .real shape) :
    Values (FP.Value .fp32) .real (TileShape.reduceShape shape ax keepDims) := by
  cases keepDims
  · exact fun out => (List.finRange (TileShape.axisDim shape ax)).foldl
      (fun a k => FP.add cfg a (x (TileShape.insertAxisIndex shape ax out k))) (FP.Value.zero .fp32)
  · exact fun out => (List.finRange (TileShape.axisDim shape ax)).foldl
      (fun a k => FP.add cfg a (x (TileShape.replaceAxisIndex shape ax out k))) (FP.Value.zero .fp32)

/-- In this source, `.real` is the erased carrier of an explicitly fp32
compute register. The adapter binds its storage/compute bit-preserving casts;
it does not identify arbitrary mathematical real-to-float casts with identity. -/
def backend (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  { FP.Software.scalarFP32 rest cfg with
    cast := fun p src dst x =>
      if (p = some .fp32 ∧ src = .fp32 ∧ dst = .real) ∨
         ((p = none ∨ p = some .fp32) ∧ src = .real ∧ dst = .fp32) then x
      else (FP.Software.scalarFP32 rest cfg).cast p src dst x
    reduceSum := fun p _ ax keep xs =>
      if p = some .fp32 then sequentialSum cfg ax keep xs
      else rest.reduceSum p ax keep xs }

def readWord [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32))
    (r : RegionName) (off : Nat) (active : Prop) [Decidable active] : FP.Value .fp32 :=
  if active then (s.mem r off).read .fp32 else FP.Value.zero .fp32

def roundedAccumulated [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) (i : TileIndex [p.blockRows,p.blockColumns]) : FP.Value .fp32 :=
  if 0 < p.columns then FP.add cfg (FP.Value.zero .fp32)
    (FP.mul cfg
      (readWord s p.a (rowIndex p s i.1 * p.strideRow + i.2.1.val * p.strideColumn)
        (rowIndex p s i.1 < p.rows ∧ i.2.1.val < p.columns))
      (readWord s p.b (i.2.1.val * p.strideVector) (i.2.1.val < p.columns)))
  else FP.Value.zero .fp32

def roundedOutput [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) (i : TileIndex [p.blockRows,1]) : FP.Value .fp32 :=
  sequentialSum cfg ⟨1, by simp⟩ Bool.false (roundedAccumulated cfg p s) (i.1,PUnit.unit)

theorem accumulated_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) :
    accumulated (backend rest cfg) .fp32 p s = roundedAccumulated cfg p s := by
  funext i
  simp [accumulated, inputValue, roundedAccumulated, readWord,
    backend, FP.Software.scalarFP32, FP.Software.fp32Context, toFloat]

theorem output_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (i : TileIndex [p.blockRows,1]) :
    outputValues (backend rest cfg) .fp32 p s i = roundedOutput cfg p s i := by
  unfold outputValues
  rw [accumulated_software]
  simp [roundedOutput, backend]
#axiomsClean accumulated_software
#axiomsClean output_software
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
