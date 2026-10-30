import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivTriton.Contiguous
open VeriTile.Torch
noncomputable section

/-- Observations of the actual Tensor and its complete untyped storage. -/
structure Tensor where
  metadata : TensorMetadata
  storageBase : Nat
  storageBytes : Nat
  deriving DecidableEq

inductive Argument where
  | nonTensor
  | tensor (value : Tensor)
  deriving DecidableEq

structure TensorContract (t : Tensor) : Prop where
  layout : t.metadata.layout = .strided
  dtype : t.metadata.dtype = .floating .fp32
  cuda : t.metadata.device.kind = "cuda"
  contiguous : t.metadata.isContiguous = Bool.true
  noNeg : t.metadata.isNeg = Bool.false
  alignment : t.metadata.numel ≠ 0 → t.metadata.dataPtr.toNat % 4 = 0

structure PairContract (a b : Tensor) : Prop where
  left : TensorContract a
  right : TensorContract b
  device : b.metadata.device = a.metadata.device
  leftRank : a.metadata.shape.length = 2
  rightRank : b.metadata.shape.length = 2
  shape : b.metadata.shape = a.metadata.shape

def StorageSeparate (a b : Tensor) : Prop :=
  a.storageBase + a.storageBytes ≤ b.storageBase ∨
  b.storageBase + b.storageBytes ≤ a.storageBase

structure BackwardContract (a b g : Tensor) : Prop extends PairContract a b where
  gradient : TensorContract g
  gradientDevice : g.metadata.device = a.metadata.device
  gradientShape : g.metadata.shape = [] ∨ g.metadata.shape = a.metadata.shape
  separate : a.metadata.numel ≠ 0 → StorageSeparate a b ∧ StorageSeparate a g

abbrev ForwardContract := PairContract

def validMode (mode : String) : Prop :=
  mode = "none" ∨ mode = "sum" ∨ mode = "mean" ∨ mode = "batchmean"

/-- Ordered checks shared by both Python entries. No value-domain guard. -/
def checkTensorList (ts : List Tensor) : Except String Unit := by
  classical
  exact
    if ¬ (∀ t ∈ ts, t.metadata.layout = .strided) then .error "layout"
    else if ¬ (∀ t ∈ ts, t.metadata.dtype = .floating .fp32) then .error "dtype"
    else if ¬ (∀ t ∈ ts, t.metadata.device.kind = "cuda") then .error "cuda"
    else if ¬ (match ts with | [] => True | a::rest => ∀ b ∈ rest, b.metadata.device = a.metadata.device) then .error "device"
    else if ¬ (∀ t ∈ ts, t.metadata.isContiguous = Bool.true) then .error "contiguous"
    else if ¬ (∀ t ∈ ts, t.metadata.isNeg = Bool.false) then .error "negative"
    else if ¬ (∀ t ∈ ts, t.metadata.numel ≠ 0 → t.metadata.dataPtr.toNat % 4 = 0) then .error "alignment"
    else .ok ()

def checkPair (a b : Tensor) : Except String Unit := by
  classical
  exact if ¬ (a.metadata.shape.length = 2 ∧ b.metadata.shape.length = 2) then .error "rank"
    else if b.metadata.shape ≠ a.metadata.shape then .error "shape" else .ok ()

def checkForward (a b : Argument) (flag : Option Bool) (mode : String) : Except String Unit := by
  classical
  exact match a,b with
  | .tensor a,.tensor b => do
    checkTensorList [a,b]
    checkPair a b
    if flag.isNone then .error "flag"
    else if ¬ validMode mode then .error "reduction" else .ok ()
  | _,_ => .error "tensor"

def checkBackward (a b g : Argument) (flag : Option Bool) : Except String Unit := by
  classical
  exact match a,b,g with
  | .tensor a,.tensor b,.tensor g => do
    checkTensorList [a,b,g]
    checkPair a b
    if ¬ (g.metadata.shape = [] ∨ g.metadata.shape = a.metadata.shape) then .error "gradient shape"
    else if flag.isNone then .error "flag"
    else if ¬ (a.metadata.numel ≠ 0 → StorageSeparate a b ∧ StorageSeparate a g) then .error "alias"
    else .ok ()
  | _,_,_ => .error "tensor"

def AdmittedForward (a b : Argument) (flag : Option Bool) (mode : String) : Prop :=
  match a,b with
  | .tensor a,.tensor b => ForwardContract a b ∧ flag.isSome = Bool.true ∧ validMode mode
  | _,_ => False

def AdmittedBackward (a b g : Argument) (flag : Option Bool) : Prop :=
  match a,b,g with
  | .tensor a,.tensor b,.tensor g => BackwardContract a b g ∧ flag.isSome = Bool.true
  | _,_,_ => False

end
end VeriTile.Bench.Repairs.KldivTriton.Contiguous
