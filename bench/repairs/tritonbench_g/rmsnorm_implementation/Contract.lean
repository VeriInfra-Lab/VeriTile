import VeriTile.Torch.Tensor
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace VeriTile.Bench.Repairs.RmsnormImplementation
open VeriTile.Torch

/-- Independent mathematical RMSNorm formula on logical row coordinates.
Real is the specification; no native FP32-to-Real equality is asserted. -/
noncomputable def mathematical (n : Nat) (eps : ℝ) (x w : Nat → ℝ) (k : Nat) : ℝ :=
  (x k / Real.sqrt ((∑ j : Fin n, (x j.val)^2)/(n : ℝ)+eps)) * w k

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
def padded (x : TensorInfo) : Nat := (width x+4095)/4096*4096
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

end VeriTile.Bench.Repairs.RmsnormImplementation
