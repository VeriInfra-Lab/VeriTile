import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeSerial

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000

theorem inner_hits_windows (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) : NativeStores.Hits (innerStores c j s) r off ↔
      (r="h" ∧ off∈historyWrites c j) ∨ (r="ht" ∧ off∈finalWrites c j) := by
  rw [innerStores,NativeStores.inner_hits]
  constructor
  · rintro (⟨n,hn,hr,i,ho⟩|⟨enabled,hr,i,ho⟩)
    · exact Or.inl ⟨hr,ho.symm ▸ history_mem c j s n hn i⟩
    · exact Or.inr ⟨hr,ho.symm ▸ final_mem c j s enabled i⟩
  · rintro (⟨hr,hit⟩|⟨hr,hit⟩)
    · obtain ⟨n,hn,hit⟩ := List.mem_flatMap.mp hit
      obtain ⟨i,_,ho⟩ := List.mem_map.mp hit
      exact Or.inl ⟨n,List.mem_range.mp hn,hr,i,ho.symm⟩
    · by_cases enabled : c.storeFinal=Bool.true
      · simp only [finalWrites,enabled,↓reduceIte] at hit
        obtain ⟨i,_,ho⟩ := List.mem_map.mp hit
        exact Or.inr ⟨enabled,hr,i,ho.symm⟩
      · simp [finalWrites,enabled] at hit

theorem prefix_hits_windows (c : HostFP.Cumulative) (j : Job c.plans) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) : NativeStores.Hits (prefixStores c j s) r off ↔
      r="o" ∧ off∈prefixCells c j := by
  rw [prefixStores,NativeStores.prefix_hits]
  constructor
  · rintro ⟨enabled,hr,i,ho⟩
    exact ⟨hr,ho.symm ▸ prefix_mem c j s enabled i⟩
  · rintro ⟨hr,hit⟩
    by_cases enabled : NativePrefix.enabled (NativeGrid.prefixShape c) (plan j).base (probe j)
    · simp only [prefixCells,if_pos enabled] at hit
      obtain ⟨i,_,ho⟩ := List.mem_map.mp hit
      exact ⟨enabled,hr,i,ho.symm⟩
    · simp [prefixCells,enabled] at hit

theorem hits_flatMap {J : Type} (jobs : List J) (writes : J → List NativeStores.Store)
    (r : RegionName) (off : Nat) : NativeStores.Hits (jobs.flatMap writes) r off ↔
      ∃ j∈jobs,NativeStores.Hits (writes j) r off := by
  simp only [NativeStores.Hits,List.mem_flatMap]
  constructor
  · rintro ⟨w,⟨j,hj,hw⟩,hr,ho⟩
    exact ⟨j,hj,w,hw,hr,ho⟩
  · rintro ⟨j,hj,w,hw,hr,ho⟩
    exact ⟨w,⟨j,hj,hw⟩,hr,ho⟩

theorem inner_all_hits (c : HostFP.Inner) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) :
    NativeStores.Hits ((allJobs c.plans).flatMap fun j=>innerStores c j s) r off ↔
      (r="h" ∧ off∈collect (historyWrites c)) ∨ (r="ht" ∧ off∈collect (finalWrites c)) := by
  rw [hits_flatMap]
  constructor
  · rintro ⟨j,_,hit⟩
    rcases (inner_hits_windows c j s r off).mp hit with ⟨hr,hw⟩|⟨hr,hw⟩
    · exact Or.inl ⟨hr,mem_collect _ j hw⟩
    · exact Or.inr ⟨hr,mem_collect _ j hw⟩
  · rintro (⟨hr,hw⟩|⟨hr,hw⟩)
    · obtain ⟨j,hj,hit⟩ := List.mem_flatMap.mp hw
      exact ⟨j,hj,(inner_hits_windows c j s r off).mpr (Or.inl ⟨hr,hit⟩)⟩
    · obtain ⟨j,hj,hit⟩ := List.mem_flatMap.mp hw
      exact ⟨j,hj,(inner_hits_windows c j s r off).mpr (Or.inr ⟨hr,hit⟩)⟩

theorem prefix_all_hits (c : HostFP.Cumulative) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) :
    NativeStores.Hits ((allJobs c.plans).flatMap fun j=>prefixStores c j s) r off ↔
      r="o" ∧ off∈collect (prefixCells c) := by
  rw [hits_flatMap]
  constructor
  · rintro ⟨j,_,hit⟩
    obtain ⟨hr,hw⟩ := (prefix_hits_windows c j s r off).mp hit
    exact ⟨hr,mem_collect _ j hw⟩
  · rintro ⟨hr,hw⟩
    obtain ⟨j,hj,hit⟩ := List.mem_flatMap.mp hw
    exact ⟨j,hj,(prefix_hits_windows c j s r off).mpr ⟨hr,hit⟩⟩

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
