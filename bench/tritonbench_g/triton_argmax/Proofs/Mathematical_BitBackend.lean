import bench.tritonbench_g.triton_argmax.Proofs.Mathematical_TypedSource
import bench.tritonbench_g.matrix_reduction.Proofs.MachineMax

namespace VeriTile.Bench.TritonBenchG.TritonArgmax.BitBackend
open VeriTile.Triton FP.Structural TypedSource MatrixReduction.MachineMax

def indexed (storage : Storage) (dtype : FloatDType) {shape : TileShape}
    (ax : Fin shape.length) (xs : Values (Bits storage) dtype.toTileDType shape) :
    Option (Values (Bits storage) dtype.toTileDType (TileShape.eraseAxis shape ax) ×
      Values (Bits storage) .nat (TileShape.eraseAxis shape ax)) :=
  if hn : 0 < TileShape.axisDim shape ax then
    let pick := fun out => best storage
      (fun k => toFloat dtype (xs (TileShape.insertAxisIndex shape ax out k))) hn
    some ((fun out => xs (TileShape.insertAxisIndex shape ax out (pick out))),
      fun out => (pick out).val)
  else none

def backend (storage : Storage) (rest : Algebra (Bits storage)) : Algebra (Bits storage) :=
  { rest with
    negInf := negativeInfinity storage
    fp32Load := id
    cast := fun p src dst x => match storage, src, dst with
      | .fp16, .real, .fp16 =>
          if x = negativeInfinity .fp16 then negativeInfinity .fp16 else rest.cast p src dst x
      | _, _, _ => rest.cast p src dst x
    indexedMax := fun _ dtype => indexed storage dtype }
end VeriTile.Bench.TritonBenchG.TritonArgmax.BitBackend
