import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Contract
open VeriTile.Torch

/-- Ordinary Python observations; booleans record exact type predicates. -/
structure Tensor where
  objectId : Nat
  isTensor : Bool
  metadata : TensorMetadata
  conjugate : Bool
  base : Nat
  bytes : Nat
  offset : Int
  strides : List Int

def denseStrides : List Nat → List Int
  | [] => []
  | _::tail => (tail.prod:Int)::denseStrides tail

def lastOffset (t : Tensor) (shape : List Nat) : Int :=
  t.offset+((shape.zip t.strides).map fun x => ((x.1:Int)-1)*x.2).sum

def Backing (t : Tensor) (shape : List Nat) : Prop :=
  (∀ stride∈t.strides,0≤stride) ∧ 0≤t.offset ∧ t.base%4=0 ∧
  t.base+t.bytes≤2^64 ∧ (t.metadata.dataPtr.toNat:Int)=(t.base:Int)+4*t.offset ∧
  0≤lastOffset t shape ∧ lastOffset t shape<(t.bytes/4:Int)

def TensorContract (t : Tensor) (shape : List Nat) (device : Device)
    (canonical contiguous : Bool) : Prop :=
  t.isTensor=Bool.true ∧ t.metadata.shape=shape ∧
  t.metadata.dtype=.floating .fp32 ∧ t.metadata.device=device ∧ device.kind="cuda" ∧
  t.metadata.layout=.strided ∧ t.metadata.isNeg=Bool.false ∧ t.conjugate=Bool.false ∧
  (contiguous=Bool.true → t.metadata.isContiguous=Bool.true) ∧
  (canonical=Bool.true → t.strides=denseStrides shape) ∧ Backing t shape

noncomputable def tensorCheck (t : Tensor) (shape : List Nat) (device : Device)
    (canonical contiguous : Bool) : Bool := by
  classical
  exact t.isTensor && decide (t.metadata.shape=shape) &&
    decide (t.metadata.dtype=.floating .fp32) && decide (t.metadata.device=device) &&
    decide (device.kind="cuda") && decide (t.metadata.layout=.strided) &&
    !t.metadata.isNeg && !t.conjugate && (!contiguous || t.metadata.isContiguous) &&
    (!canonical || decide (t.strides=denseStrides shape)) &&
    (t.strides.all fun s => decide (0≤s)) && decide (0≤t.offset) && decide (t.base%4=0) &&
    decide (t.base+t.bytes≤2^64) &&
    decide ((t.metadata.dataPtr.toNat:Int)=(t.base:Int)+4*t.offset) &&
    decide (0≤lastOffset t shape) && decide (lastOffset t shape<(t.bytes/4:Int))

structure Dimensions where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  deriving DecidableEq

def ceil64 (n : Nat) := (n+63)/64
def tileWidth (n : Nat) := if n≤16 then 16 else if n≤32 then 32 else 64
def keyShape (d : Dimensions) := [d.B,d.H,d.T,d.K]
def valueShape (d : Dimensions) := [d.B,d.H,d.T,d.V]
def stateShape (d : Dimensions) := [d.B,d.H,d.K,d.V]
def historyShape (d : Dimensions) := [d.B,d.H,ceil64 d.T*d.K,d.V]

def ShapeContract (d : Dimensions) : Prop :=
  0<d.B ∧ 0<d.H ∧ 0<d.T ∧ 0<d.K ∧ 0<d.V ∧ d.B*d.H≤65535 ∧
  ceil64 d.T≤65535 ∧ ceil64 d.V≤65535 ∧
  ceil64 d.T*64*d.K<2^31 ∧ ceil64 d.T*64*d.V<2^31 ∧
  ceil64 d.T*d.K*d.V<2^31 ∧ ceil64 d.K*64<2^31 ∧ ceil64 d.V*64<2^31 ∧
  d.B*d.H*ceil64 d.T*d.K*d.V*4<2^63 ∧
  ceil64 d.K*d.B*d.H*d.T*d.V*4<2^63

def shapeCheck (d : Dimensions) : Bool :=
  decide (0<min d.B (min d.H (min d.T (min d.K d.V)))) && decide (d.B*d.H≤65535) &&
  decide (max (ceil64 d.T) (ceil64 d.V)≤65535) &&
  decide (max (ceil64 d.T*64*d.K) (max (ceil64 d.T*64*d.V)
    (max (ceil64 d.T*d.K*d.V) (max (ceil64 d.K*64) (ceil64 d.V*64))))<2^31) &&
  decide (max (d.B*d.H*ceil64 d.T*d.K*d.V)
    (ceil64 d.K*d.B*d.H*d.T*d.V)*4<2^63)

structure Scalar where
  exactFloat : Bool
  finite : Bool
  value : ℝ

def ScaleContract (s : Scalar) : Prop :=
  s.exactFloat=Bool.true ∧ s.finite=Bool.true ∧ |s.value|≤340282346638528859811704183484516925440
noncomputable def scaleCheck (s : Scalar) : Bool := by
  classical
  exact s.exactFloat && s.finite && decide (|s.value|≤340282346638528859811704183484516925440)

structure Forward where
  dims : Dimensions
  q : Tensor
  k : Tensor
  v : Tensor
  initial : Option Tensor
  finalIsBool : Bool
  final : Bool
  checkpointIsInt : Bool
  checkpoint : Int
  scale : Scalar

def ForwardContract (a : Forward) : Prop :=
  ShapeContract a.dims ∧
  TensorContract a.k (keyShape a.dims) a.k.metadata.device Bool.true Bool.true ∧
  TensorContract a.q (keyShape a.dims) a.k.metadata.device Bool.true Bool.true ∧
  TensorContract a.v (valueShape a.dims) a.k.metadata.device Bool.true Bool.true ∧
  (∀ t∈a.initial.toList,TensorContract t (stateShape a.dims) a.k.metadata.device Bool.true Bool.true) ∧
  a.finalIsBool=Bool.true ∧ a.checkpointIsInt=Bool.true ∧
  (a.checkpoint=0 ∨ a.checkpoint=1) ∧ ScaleContract a.scale

noncomputable def forwardCheck (a : Forward) : Bool :=
  shapeCheck a.dims &&
  tensorCheck a.k (keyShape a.dims) a.k.metadata.device Bool.true Bool.true &&
  tensorCheck a.q (keyShape a.dims) a.k.metadata.device Bool.true Bool.true &&
  tensorCheck a.v (valueShape a.dims) a.k.metadata.device Bool.true Bool.true &&
  (a.initial.toList.all fun t => tensorCheck t (stateShape a.dims) a.k.metadata.device Bool.true Bool.true) &&
  a.finalIsBool && a.checkpointIsInt && decide (a.checkpoint∈([0,1]:List Int)) && scaleCheck a.scale

structure Backward where
  forward : Forward
  incoming : Option Tensor
  terminal : Option Tensor
  gradEnabled : Bool

def BackwardContract (a : Backward) : Prop :=
  ForwardContract a.forward ∧ a.gradEnabled=Bool.false ∧
  (∀ t∈a.incoming.toList,TensorContract t (valueShape a.forward.dims)
    a.forward.k.metadata.device Bool.false Bool.false) ∧
  (∀ t∈a.terminal.toList,a.forward.final=Bool.true ∧ TensorContract t
    (stateShape a.forward.dims) a.forward.k.metadata.device Bool.false Bool.false)

noncomputable def backwardCheck (a : Backward) : Bool :=
  forwardCheck a.forward && !a.gradEnabled &&
  (a.incoming.toList.all fun t => tensorCheck t (valueShape a.forward.dims)
    a.forward.k.metadata.device Bool.false Bool.false) &&
  (a.terminal.toList.all fun t => a.forward.final && tensorCheck t
    (stateShape a.forward.dims) a.forward.k.metadata.device Bool.false Bool.false)

def disjoint (a b : Tensor) : Prop := a.base+a.bytes≤b.base ∨ b.base+b.bytes≤a.base
def overlapCheck (a b : Tensor) : Bool := decide (a.base<b.base+b.bytes ∧ b.base<a.base+a.bytes)
end VeriTile.Bench.Repairs.ChunkRetentionOps.Contract
