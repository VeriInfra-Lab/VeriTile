import Mathlib
import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
open VeriTile.Torch
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
structure TensorInfo where
  metadata : TensorMetadata
  storage : Nat
  offset : Nat
  transposeContiguous : Bool

def rows (t : TensorInfo) := t.metadata.shape[0]?.getD 0
def cols (t : TensorInfo) := t.metadata.shape[1]?.getD 0
def bytes (t : TensorInfo) : Nat := if t.metadata.dtype=.other "int8" then 1 else if t.metadata.dtype=.other "float16" then 2 else 4
structure TensorContract (t : TensorInfo) (types : List DType) (rank : Nat) (column : Bool) : Prop where
  layout : t.metadata.layout=.strided
  rank : t.metadata.shape.length=rank
  dtype : t.metadata.dtype∈types
  cuda : t.metadata.device.kind="cuda"
  noNeg : t.metadata.isNeg=Bool.false
  dense : t.metadata.isContiguous=Bool.true ∨ column=Bool.true ∧ t.transposeContiguous=Bool.true
  alignment : 0<t.metadata.numel → t.metadata.dataPtr.toNat%bytes t=0

def tensorValid (t : TensorInfo) (types : List DType) (rank : Nat) (column : Bool) : Bool :=
  t.metadata.layout==.strided && t.metadata.shape.length==rank && types.contains t.metadata.dtype &&
  t.metadata.isCuda && !t.metadata.isNeg && (t.metadata.isContiguous || column && t.transposeContiguous) &&
  (t.metadata.numel==0 || t.metadata.dataPtr.toNat%bytes t==0)

def width (K : Nat) : Nat := 2^Nat.clog 2 K
structure RowContract (fpa : TensorInfo) : Prop where
  input : TensorContract fpa [.floating .fp32] 2 Bool.false
  launch : rows fpa=0 ∨ (0<cols fpa ∧ cols fpa≤2^20 ∧ rows fpa≤2^31-1 ∧
    4*(fpa.offset+rows fpa*max (cols fpa) 1+2*width (cols fpa))<2^63)
def rowValid (fpa : TensorInfo) : Bool :=
  tensorValid fpa [.floating .fp32] 2 Bool.false &&
  (rows fpa==0 || (0<cols fpa && cols fpa≤2^20 && rows fpa≤2^31-1 &&
    4*(fpa.offset+rows fpa*max (cols fpa) 1+2*width (cols fpa))<2^63))

structure Tile where
  BM : Nat
  BN : Nat
  BK : Nat
  S : Nat
  G : Nat
  deriving DecidableEq
def tiles : List Tile := [
  ⟨128,256,64,1,8⟩,
  ⟨64,256,32,1,8⟩,
  ⟨128,128,32,1,8⟩,
  ⟨128,64,32,1,8⟩,
  ⟨64,128,32,1,8⟩,
  ⟨128,32,32,1,8⟩,
  ⟨64,32,32,1,8⟩,
  ⟨32,64,32,1,8⟩,
  ⟨64,64,32,1,8⟩,
  ⟨64,64,64,1,8⟩,
  ⟨32,32,128,1,8⟩,
  ⟨64,64,32,1,16⟩,
  ⟨64,64,64,1,16⟩,
  ⟨32,32,128,1,16⟩,
  ⟨64,64,64,1,16⟩,
  ⟨64,64,128,1,16⟩,
  ⟨32,32,256,1,16⟩,
  ⟨128,256,64,2,8⟩,
  ⟨64,256,32,2,8⟩,
  ⟨128,128,32,2,8⟩,
  ⟨128,64,32,2,8⟩,
  ⟨64,128,32,2,8⟩,
  ⟨128,32,32,2,8⟩,
  ⟨64,32,32,2,8⟩,
  ⟨32,64,32,2,8⟩,
  ⟨64,64,32,2,8⟩,
  ⟨64,64,64,2,8⟩,
  ⟨32,32,128,2,8⟩,
  ⟨64,64,32,2,16⟩,
  ⟨64,64,64,2,16⟩,
  ⟨32,32,128,2,16⟩,
  ⟨64,64,64,2,16⟩,
  ⟨64,64,128,2,16⟩,
  ⟨32,32,256,2,16⟩
]

def gm (a : TensorInfo) (c : Tile) := (rows a+c.BM-1)/c.BM
def gn (b : TensorInfo) (c : Tile) := (cols b+c.BN-1)/c.BN
def fk (a : TensorInfo) (c : Tile) := ((cols a+c.BK*c.S-1)/(c.BK*c.S)+1)*(c.BK*c.S)
def columnMajor (b : TensorInfo) : Bool := !b.metadata.isContiguous
def bRowStride (a b : TensorInfo) : Nat := if columnMajor b then 1 else max (cols b) 1
def bColStride (a b : TensorInfo) : Nat := if columnMajor b then max (cols a) 1 else 1
def offset (t : Option TensorInfo) : Nat := (t.map TensorInfo.offset).getD 0

def singleOutput (out : Option TensorInfo) : Bool :=
  (out.map (fun t=>t.metadata.dtype)).getD (.other "float16")==.floating .fp32
def outputBytes (out : Option TensorInfo) : Nat := if singleOutput out then 4 else 2

structure TileContract (a b bs : TensorInfo) (as out : Option TensorInfo) (composed : Bool) (c : Tile) : Prop where
  dimensions : rows a+c.BM<2^63 ∧ cols b+c.BN<2^63 ∧ cols a+c.BK*c.S<2^63
  coordinates : gm a c*c.BM<2^63 ∧ gn b c*c.BN<2^63 ∧ fk a c<2^63
  groupWidth : c.G*gn b c<2^63
  grid : gm a c*gn b c≤2^31-1 ∧ c.S≤65535
  aFormed : (if composed then 0 else a.offset)+rows a*max (cols a) 1+fk a c<2^63
  bFormed : b.offset+fk a c*bRowStride a b+cols b*bColStride a b<2^63
  outFormed : outputBytes out*(offset out+gm a c*c.BM*max (cols b) 1+gn b c*c.BN)<2^63
  aScaleFormed : 4*((if composed then 0 else offset as)+rows a)<2^63
  bScaleFormed : 4*(bs.offset+cols b)<2^63

def tileValid (a b bs : TensorInfo) (as out : Option TensorInfo) (composed : Bool) (c : Tile) : Bool :=
  rows a+c.BM<2^63 && cols b+c.BN<2^63 && cols a+c.BK*c.S<2^63 &&
  gm a c*c.BM<2^63 && gn b c*c.BN<2^63 && fk a c<2^63 && c.G*gn b c<2^63 &&
  gm a c*gn b c≤2^31-1 && c.S≤65535 &&
  (if composed then 0 else a.offset)+rows a*max (cols a) 1+fk a c<2^63 &&
  b.offset+fk a c*bRowStride a b+cols b*bColStride a b<2^63 &&
  outputBytes out*(offset out+gm a c*c.BM*max (cols b) 1+gn b c*c.BN)<2^63 &&
  4*((if composed then 0 else offset as)+rows a)<2^63 && 4*(bs.offset+cols b)<2^63

def inputs (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (composed : Bool) : List TensorInfo :=
  if composed then [a,b,bs] else [a]++as.toList++[b,bs]
def MixedSeparate (ts : List TensorInfo) : Prop := ∀ u∈ts,∀ v∈ts,u.metadata.dtype=v.metadata.dtype ∨ u.storage≠v.storage
structure OutputContract (a b : TensorInfo) (ts : List TensorInfo) (out : TensorInfo) : Prop where
  tensor : TensorContract out [.other "float16",.floating .fp32] 2 Bool.false
  shape : out.metadata.shape=[rows a,cols b]
  device : out.metadata.device=a.metadata.device
  separated : ∀ t∈ts,out.storage≠t.storage

structure GemmContract (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (out : Option TensorInfo) (composed : Bool) : Prop where
  aTensor : TensorContract a [if composed then .floating .fp32 else .other "int8"] 2 Bool.false
  bTensor : TensorContract b [.other "int8"] 2 Bool.true
  bScaleTensor : TensorContract bs [.floating .fp32] 1 Bool.false
  aScale : composed=Bool.false → ∃ t,as=some t ∧ TensorContract t [.other "float16",.floating .fp32] 1 Bool.false ∧ t.metadata.shape=[rows a]
  shape : rows b=cols a ∧ bs.metadata.shape=[cols b]
  device : ∀ t∈inputs a as b bs composed,t.metadata.device=a.metadata.device
  mixedAlias : MixedSeparate (inputs a as b bs composed)
  output : ∀ t,out=some t → OutputContract a b (inputs a as b bs composed) t
  launch : rows a=0 ∨ cols b=0 ∨ ∀ c∈tiles,TileContract a b bs as out composed c

def outputValid (a b : TensorInfo) (ts : List TensorInfo) (out : TensorInfo) : Bool :=
  tensorValid out [.other "float16",.floating .fp32] 2 Bool.false && out.metadata.shape==[rows a,cols b] &&
  out.metadata.device==a.metadata.device && ts.all (fun t=>out.storage != t.storage)
def gemmValid (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo) (out : Option TensorInfo) (composed : Bool) : Bool :=
  tensorValid a [if composed then .floating .fp32 else .other "int8"] 2 Bool.false &&
  tensorValid b [.other "int8"] 2 Bool.true && tensorValid bs [.floating .fp32] 1 Bool.false &&
  (composed || (match as with | none=>Bool.false | some t=>tensorValid t [.other "float16",.floating .fp32] 1 Bool.false && t.metadata.shape==[rows a])) &&
  rows b==cols a && bs.metadata.shape==[cols b] &&
  (inputs a as b bs composed).all (fun t=>t.metadata.device==a.metadata.device) &&
  (inputs a as b bs composed).all (fun u=>(inputs a as b bs composed).all (fun v=>u.metadata.dtype==v.metadata.dtype || u.storage != v.storage)) &&
  (match out with | none=>Bool.true | some t=>outputValid a b (inputs a as b bs composed) t) &&
  (rows a==0 || cols b==0 || tiles.all (tileValid a b bs as out composed))

structure HelperContract (weight : TensorInfo) (axis : Int) : Prop where
  input : TensorContract weight [.floating .fp32] 2 Bool.false
  axes : axis=0 ∨ axis=1 ∨ axis= -1 ∨ axis= -2
  nonempty : 0<(if axis=0 ∨ axis= -2 then rows weight else cols weight)
def helperValid (weight : TensorInfo) (axis : Int) : Bool :=
  tensorValid weight [.floating .fp32] 2 Bool.false && (axis==0 || axis==1 || axis== -1 || axis== -2) &&
  0<(if axis=0 ∨ axis= -2 then rows weight else cols weight)

def ComposedContract (fpa b bs : TensorInfo) (out : Option TensorInfo) : Prop :=
  RowContract fpa ∧ GemmContract fpa none b bs out Bool.true

inductive Argument where
  | tensor (info : TensorInfo)
  | nonTensor
inductive OptionalArgument where
  | none
  | tensor (info : TensorInfo)
  | nonTensor
inductive AxisArgument where
  | integer (value : Int)
  | bool | nonInteger
inductive InputError where
  | classification | tensor | domain
  deriving DecidableEq

def checkRow (input : Argument) : Except InputError Unit :=
  match input with | .tensor t=>if rowValid t then .ok () else .error .domain | _=>.error .classification

def checkGemm (a as b bs : Argument) (out : OptionalArgument) : Except InputError Unit :=
  match a,as,b,bs,out with
  | .tensor a,.tensor as,.tensor b,.tensor bs,.none=>if gemmValid a (some as) b bs none Bool.false then .ok () else .error .domain
  | .tensor a,.tensor as,.tensor b,.tensor bs,.tensor out=>if gemmValid a (some as) b bs (some out) Bool.false then .ok () else .error .domain
  | _,_,_,_,_=>.error .classification

def checkComposed (fpa b bs : Argument) (out : OptionalArgument) : Except InputError Unit :=
  match fpa,b,bs,out with
  | .tensor fpa,.tensor b,.tensor bs,.none=>if rowValid fpa && gemmValid fpa none b bs none Bool.true then .ok () else .error .domain
  | .tensor fpa,.tensor b,.tensor bs,.tensor out=>if rowValid fpa && gemmValid fpa none b bs (some out) Bool.true then .ok () else .error .domain
  | _,_,_,_=>.error .classification

def checkHelper (weight : Argument) (axis : AxisArgument) : Except InputError Unit :=
  match weight,axis with
  | .tensor weight,.integer axis=>if helperValid weight axis then .ok () else .error .domain
  | _,_=>.error .classification
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
