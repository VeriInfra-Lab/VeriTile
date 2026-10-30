import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Domain
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Host
import VeriTile.Torch.Tensor
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
open VeriTile.Triton Whole
noncomputable section
open Classical

inductive Role where
  | q | k | v | h | g | dout | dh | dq | dk | dg
  deriving DecidableEq, Repr

def roles : List Role := [.q,.k,.v,.h,.g,.dout,.dh,.dq,.dk,.dg]
def inputRoles : List Role := [.q,.k,.v,.g,.dout,.h,.dh]
def IsInput (r : Role) := r∈inputRoles

def Role.region : Role → RegionName
  | .q => "q" | .k => "k" | .v => "v" | .h => "h" | .g => "g"
  | .dout => "do" | .dh => "dh" | .dq => "dq" | .dk => "dk" | .dg => "dg"

/-- Tensor metadata and actual stride queries. The scalar fields model the
accepted Python numeric kind and result of its explicit FP32 conversion. -/
structure Metadata where
  tensor : Role → VeriTile.Torch.TensorMetadata
  strides : Role → List Nat
  isConj : Role → Bool
  scalarKind : Bool
  roundedScale : Option (BitVec 32)

def scaleBits (p : Metadata) := p.roundedScale.getD 0
def scaleRat (p : Metadata) := (FP.Value.mk (scaleBits p) : FP.Value .fp32).toRat?
def config (p : Metadata) : Config where
  B := (p.tensor .k).shape.getD 0 0
  H := (p.tensor .k).shape.getD 1 0
  T := (p.tensor .k).shape.getD 2 0
  K := (p.tensor .k).shape.getD 3 0
  V := (p.tensor .v).shape.getLast?.getD 0
  scale := (scaleRat p).getD 0

def shape (p : Metadata) : Role → List Nat
  | .q | .k | .dq | .dk => [(config p).B,(config p).H,(config p).T,(config p).K]
  | .v | .dout => [(config p).B,(config p).H,(config p).T,(config p).V]
  | .h | .dh => [(config p).B,(config p).H,(config p).NT*(config p).K,(config p).V]
  | .g => [(config p).B,(config p).H,(config p).T]
  | .dg => [(config p).NK,(config p).B,(config p).H,(config p).T]
def base (p : Metadata) (r : Role) := (p.tensor r).dataPtr.toNat

def Nonempty (p : Metadata) : Prop := 0<(config p).B ∧ 0<(config p).H ∧ 0<(config p).T

def TensorGood (p : Metadata) (r : Role) : Prop :=
  (p.tensor r).shape=shape p r ∧ p.strides r=canonicalStrides (shape p r) ∧
  (p.tensor r).dtype=.floating .fp32 ∧ (p.tensor r).layout=.strided ∧
  (p.tensor r).device=(p.tensor .q).device ∧ (p.tensor r).device.kind="cuda" ∧
  (p.tensor r).isContiguous=Bool.true ∧ p.isConj r=Bool.false ∧ (p.tensor r).isNeg=Bool.false ∧
  (0<(p.tensor r).numel → base p r%4=0)

def Separate (p : Metadata) : Prop := Nonempty p → ∀ r s,IsInput r → IsInput s → r≠s →
  base p r+4*(p.tensor r).numel≤base p s ∨ base p s+4*(p.tensor s).numel≤base p r

def NativeGuard (p : Metadata) : Prop := Nonempty p →
  envelope (config p)<2^31 ∧ (config p).NK<2^31 ∧ (config p).NT≤65535 ∧ (config p).heads≤65535

def Pointers (p : Metadata) : Prop := Nonempty p → ∀ r,IsInput r →
  base p r+4*envelope (config p)<2^64

structure InputContract (p : Metadata) : Prop where
  ranks : (p.tensor .k).shape.length=4 ∧ (p.tensor .v).shape.length=4
  key : 0<(config p).K
  value : 0<(config p).V
  tensors : ∀ r,IsInput r → TensorGood p r
  scalarKind : p.scalarKind=Bool.true
  converted : p.roundedScale.isSome=Bool.true
  finite : (scaleRat p).isSome=Bool.true
  separate : Separate p
  guard : NativeGuard p
  pointers : Pointers p

inductive InputError where
  | tensor | rank | dimensions | metadata | scalarKind | conversion | finite | overlap | native | pointer
  deriving DecidableEq, Repr

def checkInputs (p : Metadata) : Except InputError Unit :=
  if ¬((p.tensor .k).shape.length=4 ∧ (p.tensor .v).shape.length=4) then .error .rank
  else if ¬(0<(config p).K ∧ 0<(config p).V) then .error .dimensions
  else if ¬(∀ r,IsInput r → TensorGood p r) then .error .metadata
  else if p.scalarKind≠Bool.true then .error .scalarKind
  else if p.roundedScale.isSome≠Bool.true then .error .conversion
  else if (scaleRat p).isSome≠Bool.true then .error .finite
  else if ¬Separate p then .error .overlap
  else if ¬NativeGuard p then .error .native
  else if ¬Pointers p then .error .pointer
  else .ok ()

def checkArguments (p : Option Metadata) : Except InputError Unit :=
  match p with | none => .error .tensor | some p => checkInputs p

def Admitted (p : Option Metadata) : Prop := match p with | none => False | some p => InputContract p

theorem checkInputs_iff (p : Metadata) : checkInputs p=.ok () ↔ InputContract p := by
  have fields : InputContract p ↔
      ((p.tensor .k).shape.length=4 ∧ (p.tensor .v).shape.length=4) ∧
      (0<(config p).K ∧ 0<(config p).V) ∧ (∀ r,IsInput r → TensorGood p r) ∧
      p.scalarKind=Bool.true ∧ p.roundedScale.isSome=Bool.true ∧ (scaleRat p).isSome=Bool.true ∧
      Separate p ∧ NativeGuard p ∧ Pointers p := by
    constructor
    · intro h; exact ⟨h.ranks,⟨h.key,h.value⟩,h.tensors,h.scalarKind,h.converted,h.finite,h.separate,h.guard,h.pointers⟩
    · rintro ⟨a,⟨b,c⟩,d,e,f,g,h,i,j⟩; exact ⟨a,b,c,d,e,f,g,h,i,j⟩
  rw [fields]
  unfold checkInputs
  split_ifs <;> simp_all
  cases hs : p.roundedScale <;> cases hf : scaleRat p <;> simp_all

theorem checkArguments_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p := by
  cases p <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem native_guard (p : Metadata) (h : InputContract p) (nonempty : Nonempty p) : Guard (config p) :=
  ⟨h.key,h.value,(h.guard nonempty).1,(h.guard nonempty).2.1,(h.guard nonempty).2.2.1,(h.guard nonempty).2.2.2⟩

theorem tensor_size (p : Metadata) (h : InputContract p) (r : Role) (used : IsInput r) :
    (p.tensor r).numel=extent (config p) r.region := by
  rw [VeriTile.Torch.TensorMetadata.numel,(h.tensors r used).1]
  cases r <;> simp [shape,extent,Role.region,Config.heads,Nat.mul_assoc]

theorem region_list : roles.map Role.region=regions := rfl

theorem region_injective (r s : Role) (h : r.region=s.region) : r=s := by
  cases r <;> cases s <;> simp_all [Role.region]

theorem role_list (r : Role) : r∈roles := by cases r <;> simp [roles]

#axiomsClean checkInputs_iff
#axiomsClean checkArguments_iff
#axiomsClean native_guard
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Direct
