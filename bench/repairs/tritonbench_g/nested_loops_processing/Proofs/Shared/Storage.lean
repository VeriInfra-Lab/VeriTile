import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Shared.Execution
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperAction

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open Address Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical
structure Backing where
  owner : Nat
  base : Nat
  nbytes : Nat
  finite : base+nbytes≤2^64

abbrev Key := Nat × Nat

def Backing.contains (b : Backing) (key : Key) : Prop :=
  key.1=b.owner ∧ b.base≤key.2 ∧ key.2<b.base+b.nbytes

structure Packed (shape : List Nat) where
  object : Nat
  owner : Nat
  dense : Dense shape.prod

def Packed.backing (p : Packed shape) : Backing :=
  ⟨p.owner,p.dense.base,shape.prod*4,p.dense.finite⟩

def Packed.read (api : API α Store Byte) (m : Nat → Store) (p : Packed shape) (i : Nat) : α :=
  api.read (m p.owner) (p.dense.address i)

def caller (backings : List Backing) (key : Key) : Prop :=
  ∃ b∈backings,b.contains key

def observe (api : API α Store Byte) (m : Nat → Store) (key : Key) : Byte :=
  api.byte (m key.1) key.2

def Frame (api : API α Store Byte) (backings : List Backing) (before after : Nat → Store) : Prop :=
  ∀ key,caller backings key → observe api after key=observe api before key

theorem frame_byte (api : API α Store Byte) (backings : List Backing) (before after : Nat → Store)
    (frame : Frame api backings before after) (b : Backing) (member : b∈backings)
    (i : Nat) (bound : i<b.nbytes) :
    api.byte (after b.owner) (b.base+i)=api.byte (before b.owner) (b.base+i) :=
  frame (b.owner,b.base+i) ⟨b,member,rfl,by omega,by omega⟩

def writeOwner (m : Nat → Store) (owner : Nat) (changed : Store) : Nat → Store :=
  Function.update m owner changed

theorem writeOwner_frame (api : API α Store Byte) (backings : List Backing)
    (m : Nat → Store) (owner : Nat) (after : Store)
    (localFrame : ∀ key,caller backings key → key.1=owner →
      api.byte after key.2=api.byte (m owner) key.2) :
    Frame api backings m (writeOwner m owner after) := by
  intro key hk
  by_cases eq : key.1=owner
  · change api.byte (Function.update m owner after key.1) key.2=api.byte (m key.1) key.2
    rw [eq,Function.update_self]
    exact localFrame key hk eq
  · simp only [observe,writeOwner,Function.update_of_ne eq]


end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
