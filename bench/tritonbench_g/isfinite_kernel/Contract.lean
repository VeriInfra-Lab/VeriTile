import VeriTile.Torch.Tensor
import VeriTile.Triton.Core.Ast

namespace VeriTile.Bench.Repairs.IsfiniteKernel
open VeriTile.Torch VeriTile.Triton

inductive Precision where
  | single | double
  deriving DecidableEq
def Precision.width : Precision → Nat | .single => 4 | .double => 8
def Precision.compute : Precision → ComputeDType | .single => .fp32 | .double => .fp64
def Precision.finite : Precision → FinitePrecision | .single => .fp32 | .double => .fp64

structure View where
  pointer : Int
  byteStride : Int

def View.signedAddress (v : View) (i : Nat) : Int := v.pointer+(i : Int)*v.byteStride
def View.address (v : View) (i : Nat) : Nat := (v.signedAddress i).toNat
def View.shift (v : View) (start : Nat) : View :=
  ⟨v.pointer+(start : Int)*v.byteStride,v.byteStride⟩


/-- Retained checked launch constants; host orchestration is optional. -/
def chunkSize : Nat := 2^30
def chunkCount (n : Nat) : Nat := (n+chunkSize-1)/chunkSize
def chunkStart (c : Nat) : Nat := c*chunkSize
def chunkLength (n c : Nat) : Nat := min chunkSize (n-chunkStart c)

/-- Observations from a Tensor or the original raw StridedBuffer. Malformed
objects/noninteger metadata are rejected before a well-typed record exists. -/
structure ViewMetadata where
  base : TensorMetadata
  rawView : Bool
  dtype : DType
  device : Device
  shape : List Nat
  strides : List Int
  pointer : Int
  storageBase : Nat
  capacity : Nat

def ViewMetadata.n (v : ViewMetadata) : Nat := v.shape.headD 0
def ViewMetadata.stride (v : ViewMetadata) : Int := v.strides.headD 0
def ViewMetadata.width (v : ViewMetadata) : Nat :=
  if v.dtype = .floating .fp32 then 4 else if v.dtype = .other "float64" then 8 else 1
def ViewMetadata.view (v : ViewMetadata) : View := ⟨v.pointer,v.stride*v.width⟩
def ViewMetadata.outputView (v : ViewMetadata) : View := ⟨v.pointer,v.stride⟩
def ViewMetadata.precision (v : ViewMetadata) : Precision :=
  if v.dtype = .floating .fp32 then .single else .double

def DTypeValid (output : Bool) (v : ViewMetadata) : Prop :=
  if output then v.dtype = .other "bool"
  else v.dtype = .floating .fp32 ∨ v.dtype = .other "float64"
instance (output : Bool) (v : ViewMetadata) : Decidable (DTypeValid output v) := by
  unfold DTypeValid; infer_instance

def last (pointer stride : Int) (n : Nat) := pointer+((n-1 : Nat) : Int)*stride
def EndpointBounds (v : ViewMetadata) : Prop :=
  (v.storageBase : Int) ≤ min v.pointer (last v.pointer v.view.byteStride v.n) ∧
  max v.pointer (last v.pointer v.view.byteStride v.n)+(v.width : Int) ≤ v.storageBase+v.capacity ∧
  v.storageBase+v.capacity ≤ 2^64
instance (v : ViewMetadata) : Decidable (EndpointBounds v) := by unfold EndpointBounds; infer_instance

structure ViewContract (output : Bool) (v : ViewMetadata) : Prop where
  layout : v.base.layout = .strided
  dtype : DTypeValid output v
  cuda : v.device.kind = "cuda" ∧ v.base.device = v.device
  flags : v.rawView = Bool.true ∨ v.base.isNeg = Bool.false
  rank : v.shape = [v.n] ∧ v.strides = [v.stride]
  strideWord : -(2^63 : Int) ≤ v.stride ∧ v.stride < 2^63
  alignment : 0 < v.n → v.pointer % (v.width : Int) = 0
  bounds : 0 < v.n → EndpointBounds v

inductive InputError where
  | object | metadata | layout | dtype | cuda | negative | rank | stride
  | alignment | bounds | shape | device | zeroStride | alias
  deriving DecidableEq
inductive ArgumentMetadata where
  | other | malformed | view (metadata : ViewMetadata)
instance : Coe ViewMetadata ArgumentMetadata := ⟨ArgumentMetadata.view⟩

def checkView (output : Bool) (v : ViewMetadata) : Except InputError Unit :=
  if v.base.layout != .strided then .error .layout
  else if ¬ DTypeValid output v then .error .dtype
  else if v.device.kind != "cuda" || v.base.device != v.device then .error .cuda
  else if !v.rawView && v.base.isNeg then .error .negative
  else if v.shape != [v.n] || v.strides != [v.stride] then .error .rank
  else if ¬ (-(2^63 : Int) ≤ v.stride ∧ v.stride < 2^63) then .error .stride
  else if 0 < v.n ∧ v.pointer % (v.width : Int) ≠ 0 then .error .alignment
  else if 0 < v.n ∧ ¬ EndpointBounds v then .error .bounds else .ok ()

def checkArgument (output : Bool) : ArgumentMetadata → Except InputError ViewMetadata
  | .other => .error .object | .malformed => .error .metadata
  | .view v => match checkView output v with | .error e => .error e | .ok () => .ok v

def contains (start stride : Int) (n : Nat) (address : Int) : Bool :=
  if stride = 0 then decide (0 < n ∧ address = start)
  else decide ((address-start)%stride = 0 ∧ 0 ≤ (address-start)/stride ∧ (address-start)/stride < (n : Int))

def SpanSeparated (x y : ViewMetadata) : Prop :=
  max x.pointer (last x.pointer x.view.byteStride x.n)+(x.width : Int) ≤
    min y.pointer (last y.pointer y.stride x.n) ∨
  max y.pointer (last y.pointer y.stride x.n) < min x.pointer (last x.pointer x.view.byteStride x.n)
instance (x y : ViewMetadata) : Decidable (SpanSeparated x y) := by unfold SpanSeparated; infer_instance

def disjointCheck (x y : ViewMetadata) : Bool :=
  decide (SpanSeparated x y) || (List.range x.n).all (fun i =>
    (List.range x.width).all (fun k => !(contains y.pointer y.stride x.n (x.view.signedAddress i+k))))

def PhysicalDisjoint (x y : ViewMetadata) : Prop :=
  ∀ i,i<x.n → ∀ k,k<x.width → ∀ j,j<x.n →
    x.view.signedAddress i+(k : Int) ≠ y.outputView.signedAddress j

structure InputContract (x y : ViewMetadata) : Prop where
  input : ViewContract Bool.false x
  output : ViewContract Bool.true y
  shape : x.n = y.n
  device : x.device = y.device
  outputStride : 0 < x.n → y.stride ≠ 0
  disjoint : PhysicalDisjoint x y

def checkPair (x y : ViewMetadata) : Except InputError Unit :=
  if x.n != y.n then .error .shape
  else if x.device != y.device then .error .device
  else if 0 < x.n ∧ y.stride = 0 then .error .zeroStride
  else if !(disjointCheck x y) then .error .alias else .ok ()

def checkInputs (x y : ArgumentMetadata) : Except InputError Unit := do
  let x ← checkArgument Bool.false x
  let y ← checkArgument Bool.true y
  checkPair x y

def AdmittedArguments (x y : ArgumentMetadata) : Prop :=
  match x,y with | .view x,.view y => InputContract x y | _,_ => False
end VeriTile.Bench.Repairs.IsfiniteKernel
