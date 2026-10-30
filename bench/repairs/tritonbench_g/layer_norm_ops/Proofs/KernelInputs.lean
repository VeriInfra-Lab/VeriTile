import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Contract
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelInputs
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32 Admission
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelInputs (ReadView direct actual_pointers native_address)
end Shared
open Shared
noncomputable section

inductive Canonical : List Nat → List Nat → Prop where
  | vector (n : Nat) : Canonical [n] [1]
  | matrix (m n : Nat) : Canonical [m,n] [n,1]

/-- Ordinary Tensor API observations. Canonical strides imply contiguity;
this is a layout API law, not a new Python rejection or a value hypothesis. -/
structure TensorInput where
  tensor : Tensor
  strides : List Nat
  laws : TensorLaws tensor
  canonical : ∀ shape steps,tensor.metadata.shape=shape → strides=steps →
    Canonical shape steps → tensor.metadata.isContiguous=Bool.true

def TensorInput.argument (t : TensorInput) : Argument :=
  .tensor ⟨t.tensor.metadata,t.strides,t.tensor.storageOffset⟩
def optionalArgument (t : Option TensorInput) : Argument :=
  match t with | none => .none | some t => t.argument

def checkedRead {α : Type} (read : Nat → α) (t : TensorInput)
    (shape steps : List Nat) (hc : Canonical shape steps)
    (h : TensorContract t.argument shape steps) : ReadView α read :=
  direct read t.tensor t.laws
    ⟨h.1,h.2.1,by rw [h.2.2.1],fun _ => t.canonical shape steps h.2.2.2.1 h.2.2.2.2.1 hc,
      h.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2⟩

def optionalRead {α : Type} (read : Nat → α) (t : Option TensorInput)
    (shape steps : List Nat) (hc : Canonical shape steps)
    (h : OptionalContract (optionalArgument t) shape steps) : Option (ReadView α read) :=
  match t with
  | none => none
  | some t => some (checkedRead read t shape steps hc h)

structure ForwardBinding (c : ForwardCall) where
  (x weight : TensorInput)
  (bias residual : Option TensorInput)
  xArg : x.argument=c.common.x
  weightArg : weight.argument=c.common.weight
  biasArg : optionalArgument bias=c.common.bias
  residualArg : optionalArgument residual=c.residual

structure BackwardBinding (c : BackwardCall) where
  (x weight dy rstd : TensorInput)
  (bias dres mean : Option TensorInput)
  xArg : x.argument=c.common.x
  weightArg : weight.argument=c.common.weight
  dyArg : dy.argument=c.dy
  rstdArg : rstd.argument=c.rstd
  biasArg : optionalArgument bias=c.common.bias
  dresArg : optionalArgument dres=c.dresidual
  meanArg : optionalArgument mean=c.mean

def forwardViews {α : Type} (read : Nat → α) (c : ForwardCall)
    (b : ForwardBinding c) (h : ForwardContract c) : RegionName → Option (ReadView α read) := fun r =>
  if r="x" then some (checkedRead read b.x _ _ (.matrix c.common.rows c.common.columns) (b.xArg.symm ▸ h.1.2.2.1))
  else if r="weight" then some (checkedRead read b.weight _ _ (.vector c.common.columns) (b.weightArg.symm ▸ h.1.2.2.2.1))
  else if r="bias" then optionalRead read b.bias _ _ (.vector c.common.columns) (b.biasArg.symm ▸ h.1.2.2.2.2.1)
  else if r="residual" then optionalRead read b.residual _ _ (.matrix c.common.rows c.common.columns) (b.residualArg.symm ▸ h.2.2.1)
  else none

theorem mean_optional (c : BackwardCall) (h : MeanContract c) :
    OptionalContract c.mean [c.common.rows] [1] := by
  unfold MeanContract at h
  split at h
  · cases he : c.mean <;> simp_all [Argument.present,OptionalContract]
  · cases he : c.mean <;> simp_all [OptionalContract,TensorContract]

def backwardViews {α : Type} (read : Nat → α) (c : BackwardCall)
    (b : BackwardBinding c) (h : BackwardContract c) : RegionName → Option (ReadView α read) := fun r =>
  if r="x" then some (checkedRead read b.x _ _ (.matrix c.common.rows c.common.columns) (b.xArg.symm ▸ h.1.2.2.1))
  else if r="weight" then some (checkedRead read b.weight _ _ (.vector c.common.columns) (b.weightArg.symm ▸ h.1.2.2.2.1))
  else if r="bias" then optionalRead read b.bias _ _ (.vector c.common.columns) (b.biasArg.symm ▸ h.1.2.2.2.2.1)
  else if r="dy" then some (checkedRead read b.dy _ _ (.matrix c.common.rows c.common.columns) (b.dyArg.symm ▸ h.2.1))
  else if r="rstd" then some (checkedRead read b.rstd _ _ (.vector c.common.rows) (b.rstdArg.symm ▸ h.2.2.2.2.1))
  else if r="dres" then optionalRead read b.dres _ _ (.matrix c.common.rows c.common.columns) (b.dresArg.symm ▸ h.2.2.1)
  else if r="mean" then optionalRead read b.mean _ _ (.vector c.common.rows) (b.meanArg.symm ▸ mean_optional c h.2.2.2.1)
  else none

variable {α : Type} [Inhabited α] {read : Nat → α}
def physical (views : RegionName → Option (ReadView α read)) (u : State α) : State α :=
  {u with mem:=fun r i => match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then
        .mk .fp32 (read ((dataView v.tensor v.laws (v.checked.dense rfl)).address i)) else u.mem r i}
def logical (views : RegionName → Option (ReadView α read)) (u : State α) : State α :=
  {u with mem:=fun r i => match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then .mk .fp32 (v.values i) else u.mem r i}
def value (views : RegionName → Option (ReadView α read)) (u : State α) (r : RegionName) (i : Nat) : α :=
  match views r with
  | none => (u.mem r i).read .fp32
  | some v => if i<v.tensor.metadata.numel then v.values i else (u.mem r i).read .fp32

theorem physical_eq_logical (views : RegionName → Option (ReadView α read)) (u : State α) :
    physical views u=logical views u := by
  have hm : (physical views u).mem=(logical views u).mem := by
    funext r i
    cases hv : views r with
    | none => simp [physical,logical,hv]
    | some v =>
      by_cases hi : i<v.tensor.metadata.numel
      · simp [physical,logical,hv,hi,v.value i hi]
      · simp [physical,logical,hv,hi]
  exact congrArg (fun m => ({u with mem:=m} : State α)) hm

theorem logical_read (views : RegionName → Option (ReadView α read)) (u : State α) (r : RegionName) (i : Nat) :
    ((logical views u).mem r i).read .fp32=value views u r i := by
  change (match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then Cell.mk .fp32 (v.values i) else u.mem r i).read .fp32=_
  unfold value
  cases hv : views r with
  | none => rfl
  | some v =>
    by_cases hi : i<v.tensor.metadata.numel <;> simp only [hi,ite_true,ite_false,Cell.read_same]

def Pointers (views : RegionName → Option (ReadView α read)) : Prop :=
  ∀ r v,views r=some v → 0<v.tensor.metadata.numel →
    BitVec.ofNat 64 ((dataView v.tensor v.laws (v.checked.dense rfl)).address 0)=v.tensor.dataPtr

theorem pointers (views : RegionName → Option (ReadView α read)) : Pointers views :=
  Shared.actual_pointers views

theorem checked_words (read : Nat → Word) (t : TensorInput)
    (shape steps : List Nat) (hc : Canonical shape steps)
    (h : TensorContract t.argument shape steps) (i : Nat) :
    (checkedRead read t shape steps hc h).values i=logicalValue read t.tensor t.laws i :=
  VeriTile.Bench.Repairs.FastLayernorm.KernelInputs.direct_fp32_values read t.tensor t.laws _ i

#axiomsClean physical_eq_logical
#axiomsClean logical_read
#axiomsClean pointers
#axiomsClean checked_words
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelInputs
