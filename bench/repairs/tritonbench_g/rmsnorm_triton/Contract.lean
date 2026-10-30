import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.rmsnorm_triton.Kernels
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Torch

structure TensorInfo extends TensorMetadata where
  stride : Nat → Nat
  storageOffset : Nat
  isConj : Bool
structure EpsilonInfo where
  pythonFloat : Bool
  finite : Bool
  value : ℝ
inductive Argument where
  | nonTensor
  | tensor (info : TensorInfo)

def batches (x : TensorInfo) : Nat := x.shape[0]?.getD 0
def rows (x : TensorInfo) : Nat := x.shape[1]?.getD 0
def width (x : TensorInfo) : Nat := x.shape[2]?.getD 0
def padded (x : TensorInfo) : Nat := (width x+1023)/1024*1024
def maxOffset (x t : TensorInfo) : Nat :=
  (batches x-1)*t.stride 0+(rows x-1)*t.stride 1+(padded x-1)*t.stride 2
def maxWeight (x w : TensorInfo) : Nat := (padded x-1)*w.stride 0

structure Representation (x : TensorInfo) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
structure Dense (x : TensorInfo) : Prop where
  contiguous : x.isContiguous=Bool.true
  negative : x.isNeg=Bool.false
  conjugate : x.isConj=Bool.false

structure LaunchInput (x w : TensorInfo) : Prop where
  gridX : batches x<2^31
  gridY : rows x≤65535
  loop : padded x<2^31
  strides : x.stride 0<2^31 ∧ x.stride 1<2^31 ∧ x.stride 2<2^31 ∧ w.stride 0<2^31
  offsets : maxOffset x x<2^31 ∧ maxWeight x w<2^31
  alignment : x.dataPtr.toNat%4=0 ∧ w.dataPtr.toNat%4=0
  pointers : x.dataPtr.toNat+4*maxOffset x x<2^64 ∧ w.dataPtr.toNat+4*maxWeight x w<2^64
structure InputContract (x w : TensorInfo) (eps : EpsilonInfo) : Prop where
  input : Representation x
  weight : Representation w
  device : x.device=w.device
  rank : x.shape.length=3
  positive : 0<width x
  weightShape : w.shape=[width x]
  inputDense : Dense x
  weightDense : Dense w
  epsilonType : eps.pythonFloat=Bool.true
  epsilonFinite : eps.finite=Bool.true
  epsilonPositive : 0<eps.value
  launch : x.numel≠0 → LaunchInput x w
structure LaunchOutput (x out : TensorInfo) : Prop where
  strides : out.stride 0<2^31 ∧ out.stride 1<2^31 ∧ out.stride 2<2^31
  offset : maxOffset x out<2^31
  alignment : out.dataPtr.toNat%4=0
  pointer : out.dataPtr.toNat+4*maxOffset x out<2^64

inductive InputError where
  | tensor | representation | device | shape | dense | epsilon | gridLoop | strides | offsets | alignment | pointers
  deriving DecidableEq

noncomputable def checkLaunchInput (x w : TensorInfo) : Except InputError Unit := by
  classical
  exact
    if batches x≥2^31 || rows x>65535 || padded x≥2^31 then .error .gridLoop
    else if x.stride 0≥2^31 || x.stride 1≥2^31 || x.stride 2≥2^31 || w.stride 0≥2^31 then .error .strides
    else if maxOffset x x≥2^31 || maxWeight x w≥2^31 then .error .offsets
    else if x.dataPtr.toNat%4 != 0 || w.dataPtr.toNat%4 != 0 then .error .alignment
    else if x.dataPtr.toNat+4*maxOffset x x≥2^64 || w.dataPtr.toNat+4*maxWeight x w≥2^64 then .error .pointers
    else .ok ()
noncomputable def checkTensorInput (x w : TensorInfo) (eps : EpsilonInfo) : Except InputError Unit := by
  classical
  exact
    if x.layout != .strided || x.dtype != .floating .fp32 || !x.isCuda ||
        w.layout != .strided || w.dtype != .floating .fp32 || !w.isCuda then .error .representation
    else if x.device != w.device then .error .device
    else if x.shape.length != 3 || width x≤0 || w.shape != [width x] then .error .shape
    else if !x.isContiguous || x.isNeg || x.isConj || !w.isContiguous || w.isNeg || w.isConj then .error .dense
    else if !eps.pythonFloat || !eps.finite || eps.value≤0 then .error .epsilon
    else if x.numel != 0 then checkLaunchInput x w else .ok ()
noncomputable def checkInput (x w : Argument) (eps : EpsilonInfo) : Except InputError Unit :=
  match x,w with
  | .tensor x,.tensor w => checkTensorInput x w eps
  | _,_ => .error .tensor
def Admitted (x w : Argument) (eps : EpsilonInfo) : Prop :=
  match x,w with
  | .tensor x,.tensor w => InputContract x w eps
  | _,_ => False
noncomputable def checkOutput (x out : TensorInfo) : Except InputError Unit := by
  classical
  exact
    if out.stride 0≥2^31 || out.stride 1≥2^31 || out.stride 2≥2^31 || maxOffset x out≥2^31 then .error .offsets
    else if out.dataPtr.toNat%4 != 0 || out.dataPtr.toNat+4*maxOffset x out≥2^64 then .error .pointers
    else .ok ()

/-- Row-major logical coordinate; physical strides remain independent. -/
def flatIndex (p : Parameters) (b m k : Nat) : Nat := (b*p.rows+m)*p.width+k

/-- Ordinary Tensor API binding. These are the actual objects whose metadata
was checked, with the same storage identity, offset and queried data pointer.
Coordinate laws are required only at active elements, permitting arbitrary
singleton strides and empty leading dimensions. Input/weight aliases remain
admitted. No execution result or numerical relation is a binding premise. -/
structure TensorBinding (x w out : TensorInfo) (p : Parameters) (byteBase : Nat → Nat) where
  input : FP32.Tensor
  weight : FP32.Tensor
  output : FP32.Tensor
  inputLaws : FP32.TensorLaws input
  weightLaws : FP32.TensorLaws weight
  outputLaws : FP32.TensorLaws output
  inputMetadata : input.metadata=x.toTensorMetadata
  weightMetadata : weight.metadata=w.toTensorMetadata
  outputMetadata : output.metadata=out.toTensorMetadata
  inputAllocation : input.storage.object=p.input.allocation
  weightAllocation : weight.storage.object=p.weights.allocation
  outputAllocation : output.storage.object=p.output.allocation
  inputStorage : input.storage.base=byteBase p.input.allocation
  weightStorage : weight.storage.base=byteBase p.weights.allocation
  outputStorage : output.storage.base=byteBase p.output.allocation
  inputIndex : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    input.physicalIndex (flatIndex p b m k)=
      VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.inputOffset p b m k
  weightIndex : ∀ k,k<p.width → weight.physicalIndex k=
    VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.weightOffset p k
  outputIndex : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    output.physicalIndex (flatIndex p b m k)=
      VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.outputOffset p b m k

/-- Logical read from the actual Tensor's backing allocation. Lazy flags are
rejected by InputContract, so admitted logical values equal these storage reads. -/
def tensorRead (memory : Memory α) (t : FP32.Tensor) (i : Nat) : α :=
  memory t.storage.object (t.physicalIndex i)

/-- Independent RMSNorm target in original Tensor coordinates. -/
noncomputable def mathematicalResult (p : Parameters) (x w : FP32.Tensor)
    (before : Memory ℝ) (b m k : Nat) : ℝ :=
  tensorRead before x (flatIndex p b m k) /
    Real.sqrt ((∑ j : Fin p.width, (tensorRead before x (flatIndex p b m j.val))^2) /
      (p.width : ℝ)+p.eps) * tensorRead before w k

end VeriTile.Bench.Repairs.RmsnormTriton
