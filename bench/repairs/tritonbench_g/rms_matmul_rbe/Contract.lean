import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.RmsMatmulRbe.Contract
open VeriTile.Torch

/-- Observations of the ordinary Python Tensor and untyped_storage APIs.
The tensor flag is false for non-Tensor arguments. -/
structure Tensor where
  tensor : Bool
  metadata : TensorMetadata
  (base bytes offset : Nat)
  strides : List Nat

def fp32 : DType := .floating .fp32
def fp16 : DType := .other "float16"
def int8 : DType := .other "int8"
def width (dtype : DType) : Nat := if dtype=fp32 then 4 else if dtype=fp16 then 2 else 1
def weightTypes : List DType := [fp16,int8]
def overlap (a b : Tensor) : Prop := a.base<b.base+b.bytes ∧ b.base<a.base+a.bytes
instance (a b : Tensor) : Decidable (overlap a b) := inferInstanceAs (Decidable (_ ∧ _))
def disjoint (a b : Tensor) : Prop := ¬overlap a b
def TensorAdmitted (t : Tensor) (dtypes : List DType) : Prop :=
  t.tensor=Bool.true ∧ t.metadata.layout=.strided ∧ t.metadata.dtype∈dtypes ∧
  t.metadata.device.kind="cuda" ∧ t.metadata.isContiguous=Bool.true ∧ t.metadata.isNeg=Bool.false ∧
  (0<t.metadata.numel → t.metadata.dataPtr.toNat%width t.metadata.dtype=0)
def tensorGuard (t : Tensor) (dtypes : List DType) : Bool :=
  t.tensor && decide (t.metadata.layout=.strided) && decide (t.metadata.dtype∈dtypes) &&
  decide (t.metadata.device.kind="cuda") && t.metadata.isContiguous && !t.metadata.isNeg &&
  !(decide (0<t.metadata.numel) && decide (t.metadata.dataPtr.toNat%width t.metadata.dtype≠0))

/-- Kind flags are observations of the exact Python predicates: isinstance
for start/numeric scalars, type(... ) is int for head counts. finite also
records failed Python numeric conversion (including OverflowError). -/
structure Scalars where
  (startInteger headsInteger dimInteger : Bool)
  (start heads dim : Int)
  (epsilonNumeric epsilonFinite thetaNumeric thetaFinite : Bool)
  (epsilon theta : ℝ)

def ScalarAdmitted (p : Scalars) (B M : Nat) (rotary : Bool) : Prop :=
  p.headsInteger=Bool.true ∧ p.dimInteger=Bool.true ∧ 0<p.heads ∧ 0<p.dim ∧ p.dim%2=0 ∧
  p.startInteger=Bool.true ∧ -(2^63:Int)≤p.start ∧ p.start<(2^63:Int) ∧
  (rotary=Bool.true → 0<B → 0<M → p.start+(16*((M+15)/16):Nat)-1<(2^63:Int)) ∧
  p.epsilonNumeric=Bool.true ∧ p.epsilonFinite=Bool.true ∧ 0<p.epsilon ∧
  p.thetaNumeric=Bool.true ∧ p.thetaFinite=Bool.true ∧ 0<p.theta
noncomputable def scalarGuard (p : Scalars) (B M : Nat) (rotary : Bool) : Bool := by
  classical
  exact p.headsInteger && p.dimInteger && decide (0<p.heads) && decide (0<p.dim) && decide (p.dim%2=0) &&
    p.startInteger && decide (-(2^63:Int)≤p.start) && decide (p.start<(2^63:Int)) &&
    !(rotary && decide (0<B) && decide (0<M) && decide (p.start+(16*((M+15)/16):Nat)-1≥(2^63:Int))) &&
    p.epsilonNumeric && p.epsilonFinite && decide (0<p.epsilon) &&
    p.thetaNumeric && p.thetaFinite && decide (0<p.theta)

def dim (t : Tensor) (i : Nat) := t.metadata.shape[i]?.getD 0
def SameReadFormat (reads : List Tensor) : Prop :=
  ∀ a∈reads,∀ b∈reads,(a.metadata.dtype=b.metadata.dtype ∧ a.base=b.base) ∨ disjoint a b
def readFormatGuard (reads : List Tensor) : Bool :=
  reads.all fun a => reads.all fun b => (decide (a.metadata.dtype=b.metadata.dtype) && decide (a.base=b.base)) || !decide (overlap a b)
/-- The Python generator checks each unordered pair once, in list order. -/
def pythonReadFormatGuard : List Tensor → Bool
  | [] => Bool.true
  | a::tail => (tail.all fun b => (decide (a.metadata.dtype=b.metadata.dtype) && decide (a.base=b.base)) || !decide (overlap a b)) && pythonReadFormatGuard tail

structure Inputs where
  (x qw kw vw rms k v : Tensor)
  scalars : Scalars
def Inputs.reads (i : Inputs) := [i.x,i.qw,i.kw,i.vw,i.rms]
def Inputs.others (i : Inputs) := [i.qw,i.kw,i.vw,i.rms,i.k,i.v]
def Inputs.sameFormat (i : Inputs) := [i.kw,i.vw,i.rms,i.k,i.v]

def InputContract (i : Inputs) : Prop :=
  TensorAdmitted i.x [fp32] ∧
  (∀ t∈i.others,TensorAdmitted t weightTypes) ∧
  (∀ t∈i.others,t.metadata.device=i.x.metadata.device) ∧
  (∀ t∈i.sameFormat,t.metadata.dtype=i.qw.metadata.dtype) ∧
  i.x.metadata.shape.length=3 ∧ i.qw.metadata.shape.length=2 ∧
  i.kw.metadata.shape=i.qw.metadata.shape ∧ i.vw.metadata.shape=i.qw.metadata.shape ∧
  ScalarAdmitted i.scalars (dim i.x 0) (dim i.x 1) Bool.true ∧
  0<dim i.x 2 ∧ dim i.qw 1=dim i.x 2 ∧ i.rms.metadata.shape=[dim i.x 2] ∧
  dim i.qw 0=i.scalars.heads.toNat*i.scalars.dim.toNat ∧
  i.k.metadata.numel=dim i.x 0*dim i.x 1*dim i.qw 0 ∧
  i.v.metadata.numel=dim i.x 0*dim i.x 1*dim i.qw 0 ∧
  SameReadFormat i.reads ∧ disjoint i.k i.v ∧
  (∀ t∈i.reads,disjoint i.k t ∧ disjoint i.v t)

inductive Error where
  | tensor | device | dtype | shape | scalar | alias
  deriving DecidableEq, Repr
noncomputable def checkInputs (i : Inputs) : Except Error Unit :=
  if !tensorGuard i.x [fp32] || !(i.others.all fun t => tensorGuard t weightTypes) then .error .tensor
  else if !(i.others.all fun t => decide (t.metadata.device=i.x.metadata.device)) then .error .device
  else if !(i.sameFormat.all fun t => decide (t.metadata.dtype=i.qw.metadata.dtype)) then .error .dtype
  else if i.x.metadata.shape.length≠3 ∨ i.qw.metadata.shape.length≠2 ∨
    i.kw.metadata.shape≠i.qw.metadata.shape ∨ i.vw.metadata.shape≠i.qw.metadata.shape then .error .shape
  else if !scalarGuard i.scalars (dim i.x 0) (dim i.x 1) Bool.true then .error .scalar
  else if dim i.x 2=0 ∨ dim i.qw 1≠dim i.x 2 ∨ i.rms.metadata.shape≠[dim i.x 2] ∨
    dim i.qw 0≠i.scalars.heads.toNat*i.scalars.dim.toNat ∨
    i.k.metadata.numel≠dim i.x 0*dim i.x 1*dim i.qw 0 ∨
    i.v.metadata.numel≠dim i.x 0*dim i.x 1*dim i.qw 0 then .error .shape
  else if !pythonReadFormatGuard i.reads || decide (overlap i.k i.v) ||
    (i.reads.any fun t => decide (overlap i.k t) || decide (overlap i.v t)) then .error .alias
  else .ok ()

end VeriTile.Bench.Repairs.RmsMatmulRbe.Contract
