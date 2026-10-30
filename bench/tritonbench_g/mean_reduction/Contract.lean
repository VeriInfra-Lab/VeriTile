import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_SourceTensor
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.MeanReduction
open VeriTile.Bench.TritonBenchG.MeanReduction.Whole
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev DType := VeriTile.Torch.DType

/-- Ordinary observation binding: metadata and the coordinate model describe
the same original Tensor and data pointer. It contains no reduction result. -/
structure TensorBinding (metadata : TensorMetadata) (x : Tensor rank) : Prop where
  shape : metadata.shape=List.ofFn x.shape
  device : metadata.device.index=x.device
  pointer : 0 < originalNumel x.shape → metadata.dataPtr=BitVec.ofNat 64 (4*x.base)
  addressFits : ∀ coordinate, 4*x.address coordinate+4 ≤ 2^64

inductive Axes where
  | single (axis : Int)
  | multiple (axes : List Int)
  | invalid
  deriving DecidableEq, Repr

def normalize (rank : Nat) : Axes → Option (List (Fin rank))
  | .invalid => none
  | .single d => if h : 0 < rank then some [normalizeAxis rank h d] else none
  | .multiple ds => if h : 0 < rank then some (ds.map (normalizeAxis rank h))
      else if ds=[] then some [] else none

structure InputContract (metadata : TensorMetadata) (shape : Fin rank → Nat)
    (dims : List (Fin rank)) (dtype : Option DType) : Prop where
  layout : metadata.layout = .strided
  inputDType : metadata.dtype = .floating .fp32
  outputDType : dtype = none ∨ dtype = some (.floating .fp32)
  cuda : metadata.device.kind = "cuda"
  noNeg : metadata.isNeg = Bool.false
  alignment : 0 < originalNumel shape → metadata.dataPtr.toNat % 4 = 0
  distinct : dims.Nodup
  positive : 0 < columns shape dims
  paddedRows : ((rows shape dims+7)/8)*8 < 2^31
  paddedCols : ((columns shape dims+7)/8)*8 < 2^31
  rowProduct : 0 < rows shape dims → (((rows shape dims+7)/8)*8-1)*columns shape dims < 2^31

inductive InputError where
  | tensor | layout | inputDType | outputDType | cuda | negative | alignment | axes | duplicate | empty | extent | product
  deriving DecidableEq, Repr

def checkNormalized (metadata : TensorMetadata) (shape : Fin rank → Nat)
    (dims : List (Fin rank)) (dtype : Option DType) : Except InputError Unit :=
  if metadata.layout != .strided then .error .layout
  else if metadata.dtype != .floating .fp32 then .error .inputDType
  else if dtype != none && dtype != some (.floating .fp32) then .error .outputDType
  else if !metadata.isCuda then .error .cuda
  else if metadata.isNeg then .error .negative
  else if originalNumel shape > 0 && metadata.dataPtr.toNat % 4 != 0 then .error .alignment
  else if ¬dims.Nodup then .error .duplicate
  else if columns shape dims = 0 then .error .empty
  else if ((rows shape dims+7)/8)*8 >= 2^31 || ((columns shape dims+7)/8)*8 >= 2^31 then .error .extent
  else if rows shape dims > 0 && (((rows shape dims+7)/8)*8-1)*columns shape dims >= 2^31 then .error .product
  else .ok ()

/-- Tensor classification and axis-type classification precede this typed
interface. The successful normalized result is retained in the public premise. -/
def checkInput (metadata : TensorMetadata) (shape : Fin rank → Nat) (axes : Axes)
    (dtype : Option DType) : Except InputError (List (Fin rank)) :=
  match normalize rank axes with
  | none => .error .axes
  | some dims => do
    checkNormalized metadata shape dims dtype
    return dims

def Admitted (metadata : TensorMetadata) (shape : Fin rank → Nat) (axes : Axes)
    (dtype : Option DType) (dims : List (Fin rank)) : Prop :=
  normalize rank axes = some dims ∧ InputContract metadata shape dims dtype

def checkArgument (argument : VeriTile.Torch.ArgumentMetadata) (shape : Fin rank → Nat)
    (axes : Axes) (dtype : Option DType) : Except InputError (List (Fin rank)) :=
  match argument with
  | .nonTensor => .error .tensor
  | .tensor metadata => checkInput metadata shape axes dtype

end VeriTile.Bench.Repairs.MeanReduction
