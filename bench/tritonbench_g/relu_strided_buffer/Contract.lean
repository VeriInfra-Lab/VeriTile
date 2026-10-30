import VeriTile.Torch.Tensor
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_HostPlan

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton VeriTile.Torch
open TritonBenchG.ReluStridedBuffer.HostPlan

/-- API observations from a Tensor or StridedBuffer. Malformed Python metadata
(nonintegral/negative shape, nonintegral stride/pointer, or invalid base object)
is classified before constructing this record. Storage capacity is bytes // 4. -/
structure ViewMetadata where
  buffer : Buffer
  base : TensorMetadata
  dtype : DType
  device : Device
  pointer : Int
  allocationBase : Nat
  capacity : Nat

def ViewMetadata.n (x : ViewMetadata) : Nat := x.buffer.shape.headD 0
def ViewMetadata.stride (x : ViewMetadata) : Int := x.buffer.strides.headD 0
def ViewMetadata.start (x : ViewMetadata) : Int := (x.allocationBase : Int)/4+x.buffer.base

def lastCell (start stride : Int) (n : Nat) : Int := start + ((n-1 : Nat) : Int)*stride

def EndpointBounds (x : ViewMetadata) : Prop :=
  0 ≤ min x.buffer.base (lastCell x.buffer.base x.stride x.n) ∧
  max x.buffer.base (lastCell x.buffer.base x.stride x.n) < (x.capacity : Int)

instance (x : ViewMetadata) : Decidable (EndpointBounds x) := by
  unfold EndpointBounds
  infer_instance

structure ViewContract (x : ViewMetadata) : Prop where
  layout : x.base.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ x.base.dtype = .floating .fp32
  cuda : x.device.kind = "cuda" ∧ x.base.device = x.device
  noNeg : x.base.isNeg = Bool.false
  rank : x.buffer.shape = [x.n] ∧ x.buffer.strides = [x.stride]
  extent : 0 < x.n ∧ x.n < 2^31
  strideWord : -(2^63 : Int) ≤ x.stride ∧ x.stride < 2^63
  alignment : x.allocationBase % 4 = 0
  pointer : x.pointer = (x.allocationBase : Int)+4*x.buffer.base
  bounds : EndpointBounds x

inductive InputError where
  | object | metadata | layout | dtype | cuda | negative | rank | extent | stride
  | alignment | pointer | bounds | shape | device | zeroStride | alias
  deriving DecidableEq

inductive ArgumentMetadata where
  | other | malformed | view (metadata : ViewMetadata)
instance : Coe ViewMetadata ArgumentMetadata := ⟨ArgumentMetadata.view⟩

def checkView (x : ViewMetadata) : Except InputError Unit :=
  if x.base.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || x.base.dtype != .floating .fp32 then .error .dtype
  else if x.device.kind != "cuda" || x.base.device != x.device then .error .cuda
  else if x.base.isNeg then .error .negative
  else if x.buffer.shape != [x.n] || x.buffer.strides != [x.stride] then .error .rank
  else if ¬ (0 < x.n ∧ x.n < 2^31) then .error .extent
  else if ¬ (-(2^63 : Int) ≤ x.stride ∧ x.stride < 2^63) then .error .stride
  else if x.allocationBase % 4 != 0 then .error .alignment
  else if x.pointer != (x.allocationBase : Int)+4*x.buffer.base then .error .pointer
  else if ¬ EndpointBounds x then .error .bounds else .ok ()

def checkArgument : ArgumentMetadata → Except InputError ViewMetadata
  | .other => .error .object | .malformed => .error .metadata
  | .view x => match checkView x with | .error e => .error e | .ok () => .ok x

/-- The Python arithmetic-progression membership predicate. With zero remainder,
Python floor division and Lean Euclidean division have the same exact quotient. -/
def containsCell (start stride : Int) (n : Nat) (address : Int) : Bool :=
  decide ((address-start)%stride = 0 ∧ 0 ≤ (address-start)/stride ∧ (address-start)/stride < (n : Int))

def SpanSeparated (x y : ViewMetadata) : Prop :=
  max x.start (lastCell x.start x.stride x.n) < min y.start (lastCell y.start y.stride x.n) ∨
  max y.start (lastCell y.start y.stride x.n) < min x.start (lastCell x.start x.stride x.n)

instance (x y : ViewMetadata) : Decidable (SpanSeparated x y) := by
  unfold SpanSeparated
  infer_instance

def disjointCheck (x y : ViewMetadata) : Bool :=
  decide (SpanSeparated x y) || (List.range x.n).all
    (fun i => !(containsCell y.start y.stride x.n (x.start+(i : Int)*x.stride)))

def PhysicalDisjoint (x y : ViewMetadata) : Prop :=
  ∀ i, i < x.n → ∀ j, j < x.n → x.start+(i : Int)*x.stride ≠ y.start+(j : Int)*y.stride

structure InputContract (x y : ViewMetadata) : Prop where
  input : ViewContract x
  output : ViewContract y
  shape : x.n = y.n
  device : x.device = y.device
  outputStride : y.stride ≠ 0
  disjoint : PhysicalDisjoint x y

def checkPair (x y : ViewMetadata) : Except InputError Unit :=
  if x.n != y.n then .error .shape
  else if x.device != y.device then .error .device
  else if y.stride = 0 then .error .zeroStride
  else if !(disjointCheck x y) then .error .alias else .ok ()

def checkInputs (x y : ArgumentMetadata) : Except InputError Unit := do
  let x ← checkArgument x
  let y ← checkArgument y
  checkPair x y

def AdmittedArguments (x y : ArgumentMetadata) : Prop :=
  match x,y with | .view x,.view y => InputContract x y | _,_ => False
end VeriTile.Bench.Repairs.ReluStridedBuffer
