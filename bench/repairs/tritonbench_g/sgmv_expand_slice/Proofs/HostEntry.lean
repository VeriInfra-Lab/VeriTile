import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostSnapshots

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.HostEntry
open HostSnapshots

def Element (α : Host.Role → Type) : Host.Role → Type
  | .input => α .input | .weight => α .weight | .output => α .output
  | .starts | .lengths | .adapters => Int

/-- An ordinary tensor view's local logical read interpretation, including
dtype, lazy negation and its element address. All roles read the same caller
byte state; their physical backing windows may overlap without restriction. -/
structure Descriptor (α : Host.Role → Type) (role : Host.Role) where
  dimensions : List Nat
  bytesPerElement : Nat
  positiveWidth : 0 < bytesPerElement
  window : Window
  read : Bytes → Fin dimensions.prod → Element α role

def contents {α : Host.Role → Type} (role : Host.Role) (view : Descriptor α role) (bytes : Bytes) : Payload α role := by
  cases role <;> exact List.ofFn (view.read bytes)

def atBytes {α : Host.Role → Type} (role : Host.Role) (view : Descriptor α role) (bytes : Bytes) : View α role where
  values := contents role view bytes
  dimensions := view.dimensions
  extent := by cases role <;> simp [contents,payloadLength]
  bytesPerElement := view.bytesPerElement
  positiveWidth := view.positiveWidth

def entry {α : Host.Role → Type} (views : (role : Host.Role) → Descriptor α role) (bytes : Bytes) :
    (role : Host.Role) → View α role := fun role => atBytes role (views role) bytes

def callerWindows {α : Host.Role → Type} (views : (role : Host.Role) → Descriptor α role) : List Window :=
  Host.snapshotOrder.map (fun role => (views role).window)

theorem caller_present {α : Host.Role → Type} (views : (role : Host.Role) → Descriptor α role) (role : Host.Role) :
    (views role).window ∈ callerWindows views := by
  apply List.mem_map.mpr
  refine ⟨role,?_,rfl⟩
  cases role <;> simp [Host.snapshotOrder]

/-- The target values and each returned snapshot read exactly the same entry
state. This is a local view/clone connection, with no output-result premise. -/
theorem same_call {α : Host.Role → Type} (views : (role : Host.Role) → Descriptor α role) (bytes : Bytes)
    (p : Prepared (entry views bytes) (callerWindows views)) :
    p.x.values = contents .input (views .input) bytes ∧
    p.w.values = contents .weight (views .weight) bytes ∧
    p.y.values = contents .output (views .output) bytes ∧
    p.starts.values = contents .starts (views .starts) bytes ∧
    p.lengths.values = contents .lengths (views .lengths) bytes ∧
    p.adapters.values = contents .adapters (views .adapters) bytes := same_entry_values p

#axiomsClean caller_present
#axiomsClean same_call
end VeriTile.Bench.Repairs.SgmvExpandSlice.HostEntry
