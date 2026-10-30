import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Shared.Copy

namespace VeriTile.Bench.Repairs.UniformSampling.Shared
open VeriTile.Bench.Repairs.LayernormFwdTriton Memory
noncomputable section
open scoped Classical
def copyPrefix (api : API α Store Byte) (count n owner : Nat)
    (address : Nat→Nat) (value : Nat→α) (m : Nat→Store) : Nat→Store :=
  Function.update m owner (Memory.run api ((Copy.events n address value).take count) (m owner))

theorem copyPrefix_frame (api : API α Store Byte) (count n owner : Nat)
    (address : Nat→Nat) (value : Nat→α) (m : Nat→Store) (r b : Nat)
    (outside : r≠owner ∨ ∀ i,i<n→b<address i ∨ address i+4≤b) :
    api.byte (copyPrefix api count n owner address value m r) b=api.byte (m r) b := by
  by_cases h:r=owner
  · subst r
    simp only [copyPrefix,Function.update_self]
    apply run_byte_frame
    intro e he
    have he:=List.mem_of_mem_take he
    obtain ⟨i,hi,rfl⟩:=List.mem_map.mp he
    exact (outside.resolve_left (not_not_intro rfl)) i (List.mem_range.mp hi)
  · simp [copyPrefix,Function.update_of_ne h]

theorem copyPrefix_full (api : API α Store Byte) (n owner : Nat)
    (address : Nat→Nat) (value : Nat→α) (m : Nat→Store) :
    copyPrefix api n n owner address value m=Copy.run api n owner address value m := by
  have ht : (Copy.events n address value).take n=Copy.events n address value :=
    List.take_of_length_le (by simp [Copy.events])
  simp only [copyPrefix,Copy.run,ht]


#axiomsClean copyPrefix_frame
#axiomsClean copyPrefix_full
end
end VeriTile.Bench.Repairs.UniformSampling.Shared
