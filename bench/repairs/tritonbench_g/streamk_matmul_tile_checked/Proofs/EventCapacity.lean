import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventWindows

/-! Concrete capacity witness: the public bounds obligation is satisfiable
for every original parameter value, including empty output and K=0. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section

theorem arrayWindow_capacity (name : String) (region size : Nat) (sizes : Nat → Nat)
    (capacity : size ≤ sizes region) : (arrayWindow name region size).InBounds sizes := by
  intro i _
  have safe := ClosedSource.inWindow_nat _ _ (lt_of_lt_of_le i.isLt capacity)
  simpa [Window.address,arrayWindow] using safe

theorem capacity (p : Parameters) (observe : V → O) :
    (io p observe).InBounds (ClosedSource.bounds p) := by
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro port member
    change port∈[⟨.input,aWindow p⟩,⟨.input,bWindow p⟩,⟨.mem,cWindow p⟩,⟨.mem,sInitialWindow p⟩] at member
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl
    · exact arrayWindow_capacity _ _ _ _ (Nat.le_refl _)
    · exact arrayWindow_capacity _ _ _ _ (Nat.le_refl _)
    · exact arrayWindow_capacity _ _ _ _ (Nat.le_refl _)
    · exact arrayWindow_capacity _ _ _ _ (Nat.le_refl _)
  · simp [io]
  · intro port member
    have equal : port=cWindow p := List.mem_singleton.mp member
    subst port
    exact arrayWindow_capacity _ _ _ _ (Nat.le_refl _)
  · simp [io]
  · intro port member
    have equal : port=sWindow p := List.mem_singleton.mp member
    subst port
    intro i _
    have safe := ClosedSource.inWindow_nat _ _ i.isLt
    simpa [Window.address,sWindow,sInitialWindow,arrayWindow,ClosedSource.bounds] using safe
  · simp [io]

#axiomsClean capacity
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
