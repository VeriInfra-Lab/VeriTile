import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Kernels

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics
open VeriTile.Torch

/-- Actual Torch metadata, including storage bounds and the two queried strides. -/
structure TensorView where
  tensor : TensorMetadata
  strides : List Int
  isConj : Bool
  requiresGrad : Bool
  storageBase : Nat
  capacity : Nat

def TensorView.rows (t : TensorView) := t.tensor.shape.headD 0
def TensorView.columns (t : TensorView) := (t.tensor.shape.drop 1).headD 0
def TensorView.rowStride (t : TensorView) := t.strides.headD 0
def TensorView.colStride (t : TensorView) := (t.strides.drop 1).headD 0
def TensorView.pointer (t : TensorView) := t.tensor.dataPtr.toNat

def TensorKind (t : TensorView) : Prop :=
  t.tensor.layout=.strided ∧ t.tensor.dtype=.floating .fp32 ∧ t.tensor.device.kind="cuda" ∧
  t.tensor.isNeg=Bool.false ∧ t.isConj=Bool.false ∧ t.tensor.shape=[t.rows,t.columns]
instance (t : TensorView) : Decidable (TensorKind t) := by unfold TensorKind; infer_instance

def TensorBounds (t : TensorView) : Prop :=
  t.strides=[t.rowStride,t.colStride] ∧
  0≤t.rowStride ∧ t.rowStride<2^63 ∧ 0≤t.colStride ∧ t.colStride<2^63 ∧
  (0<t.rows ∧ 0<t.columns → t.pointer%4=0 ∧ t.storageBase≤t.pointer ∧
    t.pointer+4*((t.rows-1)*t.rowStride.toNat+(t.columns-1)*t.colStride.toNat+1)≤t.storageBase+t.capacity ∧
    t.storageBase+t.capacity≤2^64)
instance (t : TensorView) : Decidable (TensorBounds t) := by unfold TensorBounds; infer_instance

def BlockValid (b : Int) : Prop := 1≤b ∧ b≤1024 ∧ b.toNat &&& (b.toNat-1)=0
instance (b : Int) : Decidable (BlockValid b) := by unfold BlockValid; infer_instance

def Coordinates (x : TensorView) : Prop :=
  TensorKind x ∧ x.columns=3 ∧ x.strides=[3,1] ∧ TensorBounds x
instance (x : TensorView) : Decidable (Coordinates x) := by unfold Coordinates; infer_instance

def Destination (x y : TensorView) (column : Int) : Prop :=
  TensorKind y ∧ y.rows=x.rows ∧ column.toNat+11≤y.columns ∧ y.columns<2^31 ∧
  y.strides=[y.columns,1] ∧ y.tensor.device=x.tensor.device ∧ y.requiresGrad=Bool.false ∧ TensorBounds y ∧
  (0<x.rows → x.pointer+x.rows*12≤y.pointer ∨ y.pointer+x.rows*y.columns*4≤x.pointer)
instance (x y : TensorView) (column : Int) : Decidable (Destination x y column) := by unfold Destination; infer_instance

def launchGrid (rows block : Nat) := (rows+block-1)/block

def LaunchContract (rows width block column : Nat) : Prop :=
  0<rows → launchGrid rows block<2^31 ∧ rows*3<2^63 ∧ rows*width<2^63 ∧
    (launchGrid rows block*block-1)*3+2<2^63 ∧
    (launchGrid rows block*block-1)*width+column+10<2^63
instance (rows width block column : Nat) : Decidable (LaunchContract rows width block column) := by
  unfold LaunchContract;infer_instance

def ForwardContract (x : TensorView) (y : Option TensorView) (block column : Int) : Prop :=
  Coordinates x ∧ BlockValid block ∧ 0≤column ∧
  match y with
  | none => column=0 ∧ LaunchContract x.rows 11 block.toNat column.toNat
  | some y => Destination x y column ∧ LaunchContract x.rows y.columns block.toNat column.toNat
instance (x : TensorView) (y : Option TensorView) (block column : Int) : Decidable (ForwardContract x y block column) := by
  unfold ForwardContract; cases y <;> infer_instance

def UpstreamContract (x u : TensorView) (width : Nat) : Prop :=
  TensorKind u ∧ u.tensor.shape=[x.rows,width] ∧ u.tensor.device=x.tensor.device ∧ TensorBounds u
instance (x u : TensorView) (width : Nat) : Decidable (UpstreamContract x u width) := by unfold UpstreamContract; infer_instance

inductive TensorArgument where
  | other | tensor (value : TensorView)
inductive IntegerArgument where
  | other | integer (value : Int)
inductive GuardError where
  | object | metadata | coordinates | block | column | output | upstream
  deriving DecidableEq

def tensorCheck (t : TensorView) : Except GuardError Unit :=
  if ¬TensorKind t then .error .metadata
  else if ¬TensorBounds t then .error .metadata else .ok ()

def forwardCheck (x : TensorArgument) (y : Option TensorArgument)
    (block column : IntegerArgument) : Except GuardError Unit :=
  match x,block,column with
  | .tensor x,.integer b,.integer c =>
    if ¬Coordinates x then .error .coordinates
    else if ¬BlockValid b then .error .block
    else if c<0 then .error .column
    else match y with
      | none => if c≠0 then .error .column
        else if ¬LaunchContract x.rows 11 b.toNat c.toNat then .error .block else .ok ()
      | some (.tensor y) => if ¬Destination x y c then .error .output
        else if ¬LaunchContract x.rows y.columns b.toNat c.toNat then .error .block else .ok ()
      | some .other => .error .object
  | _,_,_ => .error .object

def backwardCheck (x : TensorView) (u : TensorArgument) (width : Nat) : Except GuardError Unit :=
  match u with
  | .other => .error .object
  | .tensor u => if ¬UpstreamContract x u width then .error .upstream else .ok ()

def AdmittedForward (x : TensorArgument) (y : Option TensorArgument)
    (block column : IntegerArgument) : Prop :=
  match x,block,column,y with
  | .tensor x,.integer b,.integer c,none => ForwardContract x none b c
  | .tensor x,.integer b,.integer c,some (.tensor y) => ForwardContract x (some y) b c
  | _,_,_,_ => False

def AdmittedBackward (x : TensorView) (u : TensorArgument) (width : Nat) : Prop :=
  match u with | .tensor u => UpstreamContract x u width | .other => False

end VeriTile.Bench.Repairs.FifthOrderSphHarmonics
