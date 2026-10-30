import VeriTile.Torch.FP32
import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_Contract
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Torch
structure TensorInfo extends TensorMetadata where
  isConj : Bool
structure EpsilonInfo where
  pythonFloat : Bool
  finite : Bool
  value : ℝ
  /-- The actual runtime FP32 argument word, separate from the Real annotation. -/
  runtimeWord : BitVec 32
structure NormalizedInfo where
  valid : Bool
  dimensions : List Nat
inductive Argument where
  | nonTensor
  | tensor (info : TensorInfo)

def dim (x : TensorInfo) (n : List Nat) : Nat := x.shape.length-n.length
def rows (x : TensorInfo) (n : List Nat) : Nat := (x.shape.take (dim x n)).prod
def columns (n : List Nat) : Nat := n.prod
def block (n : Nat) : Nat :=
  if n=0 then 0 else
    let x := n-1
    let x := x ||| (x >>> 1)
    let x := x ||| (x >>> 2)
    let x := x ||| (x >>> 4)
    let x := x ||| (x >>> 8)
    let x := x ||| (x >>> 16)
    let x := x ||| (x >>> 32)
    x+1
def rowMax (x : TensorInfo) (n : List Nat) : Nat := (rows x n-1)*columns n
def paddedMax (x : TensorInfo) (n : List Nat) : Nat := rowMax x n+(block (columns n)-1)
structure Representation (x : TensorInfo) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
structure InputContract (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo) : Prop where
  input : Representation x
  weight : Representation w
  device : x.device=w.device
  normalizedType : n.valid=Bool.true
  rank : n.dimensions.length≤x.shape.length
  suffix : x.shape.drop (dim x n.dimensions)=n.dimensions
  weightShape : w.shape=n.dimensions
  positive : 0<columns n.dimensions
  flags : x.isNeg=Bool.false ∧ x.isConj=Bool.false ∧ w.isNeg=Bool.false ∧ w.isConj=Bool.false
  epsilonType : eps.pythonFloat=Bool.true
  epsilonFinite : eps.finite=Bool.true
  epsilonPositive : 0<eps.value
  rowsBound : rows x n.dimensions<2^31
  columnsBound : columns n.dimensions<2^31
  blockBound : block (columns n.dimensions)<2^31
  rowProducts : rowMax x n.dimensions<2^31
structure Pointers where
  input : BitVec 64
  weight : BitVec 64
  output : BitVec 64
structure PointerContract (x : TensorInfo) (n : List Nat) (p : Pointers) : Prop where
  aligned : p.input.toNat%4=0 ∧ p.weight.toNat%4=0 ∧ p.output.toNat%4=0
  inputEnd : p.input.toNat+4*paddedMax x n<2^64
  weightEnd : p.weight.toNat+4*(block (columns n)-1)<2^64
  outputEnd : p.output.toNat+4*paddedMax x n<2^64
inductive InputError where
  | tensor | representation | device | normalized | shape | flags | epsilon | dimensions | offsets | alignment | pointers
  deriving DecidableEq
noncomputable def checkTensorInput (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo) : Except InputError Unit := by
  classical
  exact
    if x.layout != .strided || x.dtype != .floating .fp32 || !x.isCuda ||
        w.layout != .strided || w.dtype != .floating .fp32 || !w.isCuda then .error .representation
    else if x.device != w.device then .error .device
    else if !n.valid then .error .normalized
    else if n.dimensions.length>x.shape.length || x.shape.drop (dim x n.dimensions) != n.dimensions ||
        w.shape != n.dimensions || columns n.dimensions≤0 then .error .shape
    else if x.isNeg || x.isConj || w.isNeg || w.isConj then .error .flags
    else if !eps.pythonFloat || !eps.finite || eps.value≤0 then .error .epsilon
    else if rows x n.dimensions≥2^31 || columns n.dimensions≥2^31 then .error .dimensions
    else if block (columns n.dimensions)≥2^31 || rowMax x n.dimensions≥2^31 then .error .offsets
    else .ok ()
noncomputable def checkInput (x w : Argument) (n : NormalizedInfo) (eps : EpsilonInfo) : Except InputError Unit :=
  match x,w with
  | .tensor x,.tensor w => checkTensorInput x w n eps
  | _,_ => .error .tensor
def Admitted (x w : Argument) (n : NormalizedInfo) (eps : EpsilonInfo) : Prop :=
  match x,w with
  | .tensor x,.tensor w => InputContract x w n eps
  | _,_ => False
noncomputable def checkPointers (x : TensorInfo) (n : List Nat) (p : Pointers) : Except InputError Unit := by
  classical
  exact
    if p.input.toNat%4 != 0 || p.weight.toNat%4 != 0 || p.output.toNat%4 != 0 then .error .alignment
    else if p.input.toNat+4*paddedMax x n≥2^64 || p.weight.toNat+4*(block (columns n)-1)≥2^64 ||
        p.output.toNat+4*paddedMax x n≥2^64 then .error .pointers
    else .ok ()

/-- The actual Torch Tensor and its API-certified backing storage. Queried
strides bind the same logical index map; zero/broadcast strides and storage
offsets are allowed. No original-input contiguity check is introduced. -/
structure ActualTensor where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  strides : List Nat
  isConj : Bool
  indices : ∀ i, i < tensor.metadata.numel → tensor.physicalIndex i =
    tensor.storageOffset + VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole.stridedOffset
      tensor.metadata.shape strides i

def ActualTensor.info (t : ActualTensor) : TensorInfo := ⟨t.tensor.metadata,t.isConj⟩
def ActualTensor.view (t : ActualTensor) : VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole.TensorView where
  metadata := t.tensor.metadata
  strides := t.strides
  allocation := t.tensor.storage.object
  storageBase := t.tensor.storage.base
  storageOffset := t.tensor.storageOffset
  capacity := t.tensor.storage.elements

/-- Direct logical read of the same backing storage. The checked source
rejects lazy flags before its ordinary contiguous materializations. -/
def ActualTensor.read (t : ActualTensor)
    (memory : VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole.Memory α) (i : Nat) : α :=
  memory t.tensor.storage.object (t.tensor.physicalIndex i)

/-- Exact-real values in the original Tensor's logical order. -/
noncomputable def mathematicalResult (n : List Nat) (eps : ℝ) (x w : ActualTensor)
    (entry : VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole.Memory ℝ) (k : Nat) : ℝ :=
  VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole.realFormula (columns n) eps
    (fun j => x.read entry ((k/columns n)*columns n+j)) (w.read entry) (k%columns n)

end VeriTile.Bench.Repairs.RmsNormTriton
