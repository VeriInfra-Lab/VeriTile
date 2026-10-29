import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.UniformSampling.Shared.Copy
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Memory
noncomputable section
open scoped Classical

/-- Ordinary coordinate copy contract, instantiated explicitly as local
four-byte writes. Its source values are frozen before destination writes; this
is justified for wrapper copies by their fresh allocation separation. -/
def events (n : Nat) (address : Nat→Nat) (value : Nat→α) : List (Write α) :=
  (List.range n).map fun i=>⟨address i,value i⟩

def run (api : API α Store Byte) (n owner : Nat) (address : Nat→Nat)
    (value : Nat→α) (m : Nat→Store) : Nat→Store :=
  Function.update m owner (Memory.run api (events n address value) (m owner))

theorem readback (api : API α Store Byte) (n owner : Nat) (address : Nat→Nat)
    (value : Nat→α) (m : Nat→Store)
    (separate : ∀ i j,i<n→j<n→i≠j→Apart (address i) (address j))
    {i : Nat} (hi : i<n) :
    api.read (run api n owner address value m owner) (address i)=value i := by
  simp only [run,Function.update_self]
  apply run_readback
  · exact ⟨⟨address i,value i⟩,List.mem_map.mpr ⟨i,List.mem_range.mpr hi,rfl⟩,rfl,rfl⟩
  · intro e he
    obtain ⟨j,hj,rfl⟩:=List.mem_map.mp he
    have hj:=List.mem_range.mp hj
    by_cases h:j=i
    · subst j;exact Or.inl ⟨rfl,rfl⟩
    · exact Or.inr (separate j i hj hi h)

theorem byte_frame (api : API α Store Byte) (n owner : Nat) (address : Nat→Nat)
    (value : Nat→α) (m : Nat→Store) (r a : Nat)
    (outside : r≠owner ∨ ∀ i,i<n→a<address i ∨ address i+4≤a) :
    api.byte (run api n owner address value m r) a=api.byte (m r) a := by
  by_cases hr:r=owner
  · subst r
    have h:=outside.resolve_left (not_not_intro rfl)
    simp only [run,Function.update_self]
    apply run_byte_frame
    intro e he
    obtain ⟨i,hi,rfl⟩:=List.mem_map.mp he
    exact h i (List.mem_range.mp hi)
  · simp [run,Function.update_of_ne hr]

theorem read_frame (api : API α Store Byte) (n owner : Nat) (address : Nat→Nat)
    (value : Nat→α) (m : Nat→Store) (r a : Nat)
    (outside : r≠owner ∨ ∀ i,i<n→Apart (address i) a) :
    api.read (run api n owner address value m r) a=api.read (m r) a := by
  by_cases hr:r=owner
  · subst r
    have h:=outside.resolve_left (not_not_intro rfl)
    simp only [run,Function.update_self]
    apply run_read_frame
    intro e he
    obtain ⟨i,hi,rfl⟩:=List.mem_map.mp he
    exact h i (List.mem_range.mp hi)
  · simp [run,Function.update_of_ne hr]

#axiomsClean readback
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Shared.Copy
