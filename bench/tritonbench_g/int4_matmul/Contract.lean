import bench.tritonbench_g.int4_matmul.Proofs.HelperHost

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host HelperHost

/-- Ordinary exact-Tensor observations. Logical shape/stride/flags/storage are
in the retained descriptor; the pointer is the actual data_ptr observation. -/
structure TensorInfo where
  descriptor : Tensor
  objectID : Nat
  strided : Bool
  cuda : Bool
  rank : Nat
  dataPtr : BitVec 64

def elementBytes (d : DType) : Nat := if d=.fp16 then 2 else 4
def CanonicalStrides (t : Tensor) : Prop := t.rowStride=t.columns ∧ t.columnStride=1
def PackedStrides (t : Tensor) : Prop :=
  CanonicalStrides t ∨ (t.rowStride=transposeRowStride t.rows t.columns ∧
    t.columnStride=transposeColumnStride t.rows t.columns)
instance (t : Tensor) : Decidable (CanonicalStrides t) := inferInstanceAs (Decidable (_∧_))
instance (t : Tensor) : Decidable (PackedStrides t) := inferInstanceAs (Decidable (_∨(_∧_)))

structure TensorContract (t : TensorInfo) (dtype : DType) (packed : Bool) : Prop where
  layout : t.strided=Bool.true
  dtype : t.descriptor.dtype=dtype
  device : t.cuda=Bool.true
  rank : t.rank=2
  positive : 0<t.descriptor.rows ∧ 0<t.descriptor.columns
  offset : t.descriptor.offset=0
  flags : t.descriptor.lazyConj=Bool.false ∧ t.descriptor.lazyNeg=Bool.false ∧ t.descriptor.requiresGrad=Bool.false
  alignment : t.dataPtr.toNat%elementBytes t.descriptor.dtype=0
  strides : if packed then PackedStrides t.descriptor else CanonicalStrides t.descriptor

inductive InputError where
  | tensor | descriptor | positive | direct | alignment | stride | geometry | device | alias | bounds | scalar | finite | scale
  deriving DecidableEq

def checkTensor (t : TensorInfo) (dtype : DType) (packed : Bool) : Except InputError Unit := do
  if !t.strided || t.descriptor.dtype != dtype || !t.cuda || t.rank != 2 then throw .descriptor
  if t.descriptor.rows=0 ∨ t.descriptor.columns=0 then throw .positive
  if t.descriptor.offset != 0 || t.descriptor.lazyConj || t.descriptor.lazyNeg || t.descriptor.requiresGrad then throw .direct
  if t.dataPtr.toNat%elementBytes t.descriptor.dtype != 0 then throw .alignment
  if !(decide (CanonicalStrides t.descriptor)) && !(packed && decide (t.descriptor.rowStride=transposeRowStride t.descriptor.rows t.descriptor.columns ∧
    t.descriptor.columnStride=transposeColumnStride t.descriptor.rows t.descriptor.columns)) then throw .stride
  return ()

def packedK (weight : TensorInfo) := weight.descriptor.rows*8
def packedN (weight : TensorInfo) := weight.descriptor.columns
def packedGroup (weight scales : TensorInfo) := packedK weight/scales.descriptor.rows

def Independent (tensors : List TensorInfo) : Prop := (tensors.map (fun t=>t.descriptor.allocation)).Nodup
instance (tensors : List TensorInfo) : Decidable (Independent tensors) := inferInstanceAs (Decidable (List.Nodup _))
def SameDevice (tensors : List TensorInfo) : Prop := ∀ t∈tensors,∀ u∈tensors,t.descriptor.device=u.descriptor.device
instance (tensors : List TensorInfo) : Decidable (SameDevice tensors) := inferInstanceAs (Decidable (∀ t∈tensors,∀ u∈tensors,_))

structure PackedContract (w s z : TensorInfo) : Prop where
  weight : TensorContract w .int32 Bool.true
  scales : TensorContract s .fp32 Bool.true
  zeros : TensorContract z .int32 Bool.true
  geometry : packedN w%8=0 ∧ packedK w%s.descriptor.rows=0 ∧ s.descriptor.columns=packedN w ∧
    z.descriptor.rows=s.descriptor.rows ∧ z.descriptor.columns=packedN w/8
  device : SameDevice [w,s,z]
  independent : Independent [w,s,z]
  bounds : packedK w*packedN w<2^31 ∧ s.descriptor.rows*s.descriptor.columns<2^31

def checkPacked (weight scales zeros : TensorInfo) : Except InputError Unit := do
  checkTensor weight .int32 Bool.true
  checkTensor scales .fp32 Bool.true
  if packedN weight%8 != 0 || packedK weight%scales.descriptor.rows != 0 || scales.descriptor.columns != packedN weight then throw .geometry
  checkTensor zeros .int32 Bool.true
  if zeros.descriptor.rows != scales.descriptor.rows || zeros.descriptor.columns != packedN weight/8 then throw .geometry
  if !(decide (SameDevice [weight,scales,zeros])) then throw .device
  if !(decide (Independent [weight,scales,zeros])) then throw .alias
  if packedK weight*packedN weight≥2^31 ∨ scales.descriptor.rows*scales.descriptor.columns≥2^31 then throw .bounds
  return ()

def matrixShape (input weight scales : TensorInfo) : Shape :=
  ⟨input.descriptor.rows,packedN weight,packedK weight,packedGroup weight scales⟩
def formed (s : Shape) (weight : TensorInfo) : List Nat :=
  let kp:=kernelKGuard s
  let mp:=ceil s.m 128*128
  let np:=ceil s.n 128*128
  [s.m*s.k+kp,(kp/8+1)*s.n,(kp/s.group+1)*s.n,(kp/s.group+1)*(s.n/8),mp*s.n+np,s.m*s.n,
    mp,np,kp,mp*np,1024*weight.descriptor.rowStride]

def matrixInputs (input weight scales zeros : TensorInfo) (out : Option TensorInfo) :=
  [input,weight,scales,zeros]++out.toList
structure OutputContract (input weight out : TensorInfo) : Prop where
  tensor : TensorContract out .fp32 Bool.false
  shape : out.descriptor.rows=input.descriptor.rows ∧ out.descriptor.columns=packedN weight

def checkOutput (input weight out : TensorInfo) : Except InputError Unit := do
  checkTensor out .fp32 Bool.false
  if out.descriptor.rows != input.descriptor.rows || out.descriptor.columns != packedN weight then throw .geometry
  return ()

structure MatmulContract (a weight scales zeros : TensorInfo) (g : Int) (out : Option TensorInfo) : Prop where
  packed : PackedContract weight scales zeros
  input : TensorContract a .fp32 Bool.false
  group : 0<g ∧ g.toNat=packedGroup weight scales
  inner : a.descriptor.columns=packedK weight
  output : ∀ t,out=some t → OutputContract a weight t
  device : SameDevice (matrixInputs a weight scales zeros out)
  independent : Independent (matrixInputs a weight scales zeros out)
  bounds : ∀ n∈formed (matrixShape a weight scales) weight,n<2^31

def checkMatmul (input weight scales zeros : TensorInfo) (group : Int) (out : Option TensorInfo) : Except InputError Unit := do
  checkPacked weight scales zeros
  checkTensor input .fp32 Bool.false
  if group≤0 ∨ group.toNat != packedGroup weight scales ∨ input.descriptor.columns != packedK weight then throw .geometry
  match out with
  | none=>pure ()
  | some provided=>checkOutput input weight provided
  if !(decide (SameDevice (matrixInputs input weight scales zeros out))) then throw .device
  if !(decide (Independent (matrixInputs input weight scales zeros out))) then throw .alias
  if (formed (matrixShape input weight scales) weight).any (fun n=>decide (2^31≤n)) then throw .bounds
  return ()

structure QuantMetadataContract (w : TensorInfo) (g tpRank : Int) : Prop where
  weight : TensorContract w .fp32 Bool.false
  group : 0<g ∧ w.descriptor.rows%g.toNat=0
  packed : w.descriptor.rows%8=0 ∧ w.descriptor.columns%8=0
  rank : tpRank=(w.descriptor.device:Int)
  bounds : w.descriptor.rows*w.descriptor.columns<2^31

def checkQuantMetadata (weight : TensorInfo) (group rank : Int) : Except InputError Unit := do
  checkTensor weight .fp32 Bool.false
  if group≤0 ∨ weight.descriptor.rows%group.toNat != 0 ∨ weight.descriptor.rows%8 != 0 ∨ weight.descriptor.columns%8 != 0 then throw .geometry
  if rank != (weight.descriptor.device:Int) then throw .device
  if weight.descriptor.rows*weight.descriptor.columns≥2^31 then throw .bounds
  return ()

inductive Argument where
  | tensor (value : TensorInfo)
  | other
inductive ScalarArgument where
  | integer (value : Int)
  | other
inductive OutputArgument where
  | none
  | tensor (value : TensorInfo)
  | other

def checkMatmulArguments (input weight scales zeros : Argument) (group : ScalarArgument) (out : OutputArgument) : Except InputError Unit :=
  match input,weight,scales,zeros,group,out with
  | .tensor a,.tensor w,.tensor s,.tensor z,.integer g,.none=>checkMatmul a w s z g none
  | .tensor a,.tensor w,.tensor s,.tensor z,.integer g,.tensor o=>checkMatmul a w s z g (some o)
  | _,_,_,_,_,_=>.error .tensor

def AdmittedMatmul (input weight scales zeros : Argument) (group : ScalarArgument) (out : OutputArgument) : Prop :=
  match input,weight,scales,zeros,group,out with
  | .tensor a,.tensor w,.tensor s,.tensor z,.integer g,.none=>MatmulContract a w s z g none
  | .tensor a,.tensor w,.tensor s,.tensor z,.integer g,.tensor o=>MatmulContract a w s z g (some o)
  | _,_,_,_,_,_=>False

def checkPackedArguments (weight scales zeros : Argument) : Except InputError Unit :=
  match weight,scales,zeros with
  | .tensor w,.tensor s,.tensor z=>checkPacked w s z
  | _,_,_=>.error .tensor

def checkQuantArguments (weight : Argument) (group rank : ScalarArgument) : Except InputError Unit :=
  match weight,group,rank with
  | .tensor w,.integer g,.integer r=>checkQuantMetadata w g r
  | _,_,_=>.error .tensor

end VeriTile.Bench.Repairs.Int4Matmul.Direct
