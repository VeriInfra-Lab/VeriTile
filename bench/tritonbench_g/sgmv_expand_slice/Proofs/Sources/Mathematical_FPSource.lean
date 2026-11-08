import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
inductive Storage where
  | half | bfloat
  deriving DecidableEq, Repr
inductive Output where
  | half | bfloat | single
  deriving DecidableEq, Repr
def storageType : Storage → FloatDType
  | .half => .fp16 | .bfloat => .bf16
def outputType : Output → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .fp32

structure Parameters where
  (input weight output : RegionName)
  (starts lengths adapters : Region .nat)
  (columns inner inputRowStride inputInnerStride adapterStride weightColumnStride weightInnerStride
    outputRowStride outputColumnStride sliceOffset blockRows blockColumns blockInner : Nat)

end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
