import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

/-! Observations used by the ordered Python admission checks. Scalar value is
the exact Real annotation of the inspected finite Python binary64 float. -/
namespace VeriTile.Bench.Repairs.RmsnormFused
open VeriTile.Torch

structure TensorInfo extends TensorMetadata where
  storageOffset : Nat
  isConj : Bool

structure EpsilonInfo where
  pythonFloat : Bool
  finite : Bool
  value : ℝ

inductive Argument where
  | nonTensor
  | tensor (info : TensorInfo)

def width (x : TensorInfo) : Nat := x.shape.getLast?.getD 0
def rows (x : TensorInfo) : Nat := x.numel / width x

/-- min(16384, next_power_of_2(N)) for a positive feature width. -/
def block (n : Nat) : Nat :=
  if n≤1 then 1 else if n≤2 then 2 else if n≤4 then 4 else
  if n≤8 then 8 else if n≤16 then 16 else if n≤32 then 32 else
  if n≤64 then 64 else if n≤128 then 128 else if n≤256 then 256 else
  if n≤512 then 512 else if n≤1024 then 1024 else if n≤2048 then 2048 else
  if n≤4096 then 4096 else if n≤8192 then 8192 else 16384

structure Representation (x : TensorInfo) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
structure Canonical (x : TensorInfo) : Prop where
  contiguous : x.isContiguous=Bool.true
  offset : x.storageOffset=0
  negative : x.isNeg=Bool.false
  conjugate : x.isConj=Bool.false

structure InputContract (x weight : TensorInfo) (eps : EpsilonInfo) : Prop where
  input : Representation x
  weightRepresentation : Representation weight
  device : x.device=weight.device
  rank : 1≤x.shape.length
  widthPositive : 0<width x
  weightShape : weight.shape=[width x]
  inputCanonical : Canonical x
  weightCanonical : Canonical weight
  epsilonType : eps.pythonFloat=Bool.true
  epsilonFinite : eps.finite=Bool.true
  epsilonPositive : 0<eps.value
  block : width x≤block (width x)
  byteExtent : x.numel≤(2^63-1)/4
  grid : rows x≤2^31-1
  alignment : x.numel≠0 → x.dataPtr.toNat%4=0 ∧ weight.dataPtr.toNat%4=0

inductive InputError where
  | tensor | representation | device | shape | canonical | epsilon | featureLimit | extent | grid | alignment
  deriving DecidableEq

noncomputable def checkTensorInput (x weight : TensorInfo) (eps : EpsilonInfo) : Except InputError Unit := by
  classical
  exact
    if x.layout != .strided || x.dtype != .floating .fp32 || !x.isCuda ||
        weight.layout != .strided || weight.dtype != .floating .fp32 || !weight.isCuda then .error .representation
    else if x.device != weight.device then .error .device
    else if x.shape.length < 1 || width x ≤ 0 || weight.shape != [width x] then .error .shape
    else if !x.isContiguous || x.storageOffset != 0 || x.isNeg || x.isConj ||
        !weight.isContiguous || weight.storageOffset != 0 || weight.isNeg || weight.isConj then .error .canonical
    else if !eps.pythonFloat || !eps.finite || eps.value ≤ 0 then .error .epsilon
    else if width x > block (width x) then .error .featureLimit
    else if x.numel > (2^63-1)/4 then .error .extent
    else if rows x > 2^31-1 then .error .grid
    else if x.numel != 0 && (x.dataPtr.toNat%4 != 0 || weight.dataPtr.toNat%4 != 0) then .error .alignment
    else .ok ()

noncomputable def checkInput (x weight : Argument) (eps : EpsilonInfo) : Except InputError Unit :=
  match x,weight with
  | .tensor x,.tensor weight => checkTensorInput x weight eps
  | _,_ => .error .tensor

def Admitted (x weight : Argument) (eps : EpsilonInfo) : Prop :=
  match x,weight with
  | .tensor x,.tensor weight => InputContract x weight eps
  | _,_ => False

end VeriTile.Bench.Repairs.RmsnormFused
