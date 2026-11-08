import VeriTile.Triton
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open VeriTile.Triton
inductive ScalarFormat where
  | half | bfloat | single
  deriving DecidableEq, Repr
inductive LowFormat where
  | half | bfloat
  deriving DecidableEq, Repr
inductive MetadataWidth where
  | w32 | w64
  deriving DecidableEq, Repr

def ScalarFormat.dtype : ScalarFormat → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .fp32

def LowFormat.scalar : LowFormat → ScalarFormat
  | .half => .half | .bfloat => .bfloat

structure SourceABI where
  input : ScalarFormat
  weight : LowFormat
  output : ScalarFormat
  startsWidth : MetadataWidth
  lengthsWidth : MetadataWidth
  adaptersWidth : MetadataWidth
  addInputs : Bool

structure KernelArgs where
  (input weight output starts lengths adapters : RegionName)
  (columns inner inputRowStride inputInnerStride adapterStride weightColumnStride
    weightInnerStride outputRowStride outputColumnStride slice batch : Nat)

end VeriTile.Bench.Repairs.SgmvExpandSlice
