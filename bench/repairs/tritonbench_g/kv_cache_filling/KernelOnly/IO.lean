import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SharedTarget
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Finite row IO for each scheduled cache-fill job. Different jobs retain
independent entry states, as in the accepted source-family theorem. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (r : Request) : List IOChannel :=
  [⟨.int, [], "src"⟩, ⟨.int, [], "dst"⟩,
   ⟨.real, [r.dk], "key"⟩, ⟨.real, [r.dv], "value"⟩]

def cacheDType : Policy → TileDType
  | .copy => .real | .int4 | .int8 => .nat

def outputs (r : Request) : List IOChannel :=
  [⟨cacheDType r.policy, [r.ck], "keyCache"⟩,
   ⟨cacheDType r.policy, [r.cv], "valueCache"⟩,
   ⟨.real, [2], "keyMeta"⟩, ⟨.real, [2], "valueMeta"⟩]

abbrev Inputs (α : Type) (r : Request) := ChannelValues α (inputs r)
abbrev Outputs (α : Type) (r : Request) := ChannelValues α (outputs r)

def inputAddress (r : Request) (j : Job) (pid : Nat) :
    (i : Fin (inputs r).length) → TileIndex (inputs r)[i].shape → Nat
  | ⟨0, _⟩ | ⟨1, _⟩ => fun _ => pid
  | ⟨2, _⟩ => fun k => sourceRow r j * r.dk + k.1.val
  | ⟨3, _⟩ => fun k => sourceRow r j * r.dv + k.1.val
  | ⟨i + 4, h⟩ => False.elim (by simp [inputs] at h)

def outputAddress (r : Request) (j : Job) :
    (o : Fin (outputs r).length) → TileIndex (outputs r)[o].shape → Nat
  | ⟨0, _⟩ => fun k => destinationRow r j * r.ck + k.1.val
  | ⟨1, _⟩ => fun k => destinationRow r j * r.cv + k.1.val
  | ⟨2, _⟩ | ⟨3, _⟩ => fun k => destinationRow r j * 2 + k.1.val
  | ⟨i + 4, h⟩ => False.elim (by simp [outputs] at h)

def outputMask (r : Request) : (o : Fin (outputs r).length) →
    TileIndex (outputs r)[o].shape → Prop
  | ⟨0, _⟩ | ⟨1, _⟩ => fun _ => True
  | ⟨2, _⟩ | ⟨3, _⟩ => fun _ => r.policy ≠ .copy
  | ⟨i + 4, h⟩ => False.elim (by simp [outputs] at h)

def memory (α : Type) (r : Request) (j : Job) (pid : Nat) : MemoryIO α where
  inputs := inputs r
  outputs := outputs r
  scratch := []
  buffers := ["src", "dst", "key", "value", "keyCache", "valueCache", "keyMeta", "valueMeta"]
  read := fun i _ k => inputAddress r j pid i k
  inputMask := fun _ _ _ => True
  write := fun o _ k => outputAddress r j o k
  outputMask := fun o _ k => outputMask r o k
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def invocation {α : Type} (ops : QuantProfile.Ops α) (r : Request)
    (version : Bool) (pid : Nat) : Invocation α where
  kernel := Shared.kernel version r
  backend := QuantProfile.algebra ops
  arguments := []
  pids := fun _ => pid
  setPids := fun ax => ax = 0

def programIO {α : Type} (ops : QuantProfile.Ops α) (r : Request) (j : Job)
    (version : Bool) (pid : Nat) : ProgramIO α α :=
  { memory α r j pid with
    schedule := fun _ => [invocation ops r version pid]
    encode := id
    observe := fun _ _ => id }

/-- The candidate uses the same typed inputs, outputs and writable windows. -/
def candidate {α : Type} (ops : QuantProfile.Ops α) (r : Request) (j : Job) (pid : Nat) :
    ProgramIO.Variant (programIO ops r j Bool.false pid) where
  schedule := fun _ => [invocation ops r Bool.true pid]
  observe := fun _ _ => id

/-- The two decoded scheduling integers, not source execution or postconditions. -/
def inputDomain {α : Type} (r : Request) (j : Job) (v : Inputs α r) : Prop :=
  v ⟨0, by simp [inputs]⟩ PUnit.unit = (sourceRow r j : Int) ∧
  v ⟨1, by simp [inputs]⟩ PUnit.unit = (destinationRow r j : Int)

/-- The two existing complete source versions share the same decoded job
inputs, policy-specific outputs, observers and exterior footprint. -/
def original {α : Type} (ops : QuantProfile.Ops α) (r : Request) (j : Job) (pid : Nat) :=
  ProgramIO.Implementation.original (programIO ops r j Bool.false pid) (inputDomain r j)

def indexCommuted {α : Type} (ops : QuantProfile.Ops α) (r : Request) (j : Job) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (candidate ops r j pid) (inputDomain r j)

def keyRow {α : Type} [Inhabited α] (r : Request) (v : Inputs α r) (i : Nat) : α :=
  if h : i < r.dk then v ⟨2, by simp [inputs]⟩ (⟨i,h⟩,PUnit.unit) else default

def valueRow {α : Type} [Inhabited α] (r : Request) (v : Inputs α r) (i : Nat) : α :=
  if h : i < r.dv then v ⟨3, by simp [inputs]⟩ (⟨i,h⟩,PUnit.unit) else default

/-- Decode the two allowed result carriers according to the selected policy. -/
def cacheValue {α : Type} [Inhabited α] : (policy : Policy) →
    Cell α (BitVec 8) → Value α (cacheDType policy)
  | .copy, .fp32 x => x
  | .copy, .uint8 _ => (default : α)
  | .int4, .uint8 x | .int8, .uint8 x => x.toNat
  | .int4, .fp32 _ | .int8, .fp32 _ => (0 : Nat)

def floatValue {α : Type} [Inhabited α] : Cell α (BitVec 8) → α
  | .fp32 x => x | .uint8 _ => default

/-- Package independent address formulas as the four finite output channels. -/
def result {α : Type} [Inhabited α] (r : Request)
    (f : Address → Cell α (BitVec 8)) : Outputs α r
  | ⟨0, _⟩ => fun k => cacheValue r.policy (f ⟨.key,0,k.1.val⟩)
  | ⟨1, _⟩ => fun k => cacheValue r.policy (f ⟨.value,0,k.1.val⟩)
  | ⟨2, _⟩ => fun k => floatValue (α := α) (f ⟨.keyMeta,0,k.1.val⟩)
  | ⟨3, _⟩ => fun k => floatValue (α := α) (f ⟨.valueMeta,0,k.1.val⟩)
  | ⟨i + 4, h⟩ => False.elim (by simp [outputs] at h)

def target {α : Type} [Inhabited α] (ops : QuantProfile.Ops α)
    (r : Request) (v : Inputs α r) : Outputs α r :=
  result r (SharedTarget.value ops r (keyRow r v) (valueRow r v))

end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
