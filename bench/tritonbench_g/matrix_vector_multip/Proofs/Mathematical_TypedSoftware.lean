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

end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
