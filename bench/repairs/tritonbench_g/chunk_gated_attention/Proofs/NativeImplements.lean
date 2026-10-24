import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeInputBinding
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeIOSafety
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeStoreTypes

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000

theorem cell_identity (cell : Cell NativeGraph.Expr) : mapCell id cell=cell := by
  cases cell with
  | mk d v => cases d <;> rfl

theorem inner_outside (c : HostFP.Inner) (x : (innerIO c).Inputs)
    (s : State NativeGraph.Expr) (r : RegionName) (off : Nat)
    (outside : (innerIO c).Outside x r off) :
    ¬ NativeStores.Hits ((allJobs c.plans).flatMap fun j=>innerStores c j s) r off := by
  rw [inner_all_hits]
  rintro (⟨hr,member⟩|⟨hr,member⟩)
  · obtain ⟨i,hi,ho⟩ := List.getElem_of_mem member
    have other := outside.1 (0:Fin 2) (⟨i,hi⟩,PUnit.unit) trivial
    exact other.elim (fun ne=>ne hr) (fun ne=>ne ho.symm)
  · obtain ⟨i,hi,ho⟩ := List.getElem_of_mem member
    have other := outside.1 (1:Fin 2) (⟨i,hi⟩,PUnit.unit) trivial
    exact other.elim (fun ne=>ne hr) (fun ne=>ne ho.symm)

theorem prefix_outside (c : HostFP.Cumulative) (x : (prefixIO c).Inputs)
    (s : State NativeGraph.Expr) (r : RegionName) (off : Nat)
    (outside : (prefixIO c).Outside x r off) :
    ¬ NativeStores.Hits ((allJobs c.plans).flatMap fun j=>prefixStores c j s) r off := by
  rw [prefix_all_hits]
  rintro ⟨hr,member⟩
  obtain ⟨i,hi,ho⟩ := List.getElem_of_mem member
  have other := outside.1 (0:Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
  exact other.elim (fun ne=>ne hr) (fun ne=>ne ho.symm)

/-- The complete native source has the independent arithmetic graph at every
actual history/final output cell, with exact tags and exterior cell frame. -/
theorem inner_implements (c : HostFP.Inner) : (innerIO c).Implements (innerGraph c) := by
  apply ProgramIO.Implements.intro
  · intro x bounds bounded s _
    exact inner_io_safe c x bounds bounded s
  · intro x s bound
    obtain ⟨final,ran,memory⟩ := inner_run c (allJobs c.plans) s
    have writes : (allJobs c.plans).flatMap (fun j=>innerStores c j s)=innerGraphStores c x := by
      simp only [innerGraphStores,inner_binding c x s bound]
    rw [writes] at memory
    refine ⟨final,ran,?_,?_⟩
    · intro o j _
      change Fin 2 at o
      fin_cases o
      · let off := (collect (historyWrites c))[j.1]
        have hit : NativeStores.Hits (innerGraphStores c x) "h" off :=
          (inner_all_hits c (innerInput c x) "h" off).mpr
            (Or.inl ⟨rfl,List.getElem_mem _⟩)
        change mapCell id (final.mem "h" off)=
          .mk .real ((NativeStores.apply (innerGraphStores c x) emptyState.mem "h" off).read .real)
        rw [cell_identity,memory,NativeStores.at_hit _ _ emptyState.mem _ _ hit]
        exact NativeStores.apply_tag _ _ _ _ _ (inner_all_typed c (innerInput c x)).1 hit
      · let off := (collect (finalWrites c))[j.1]
        have hit : NativeStores.Hits (innerGraphStores c x) "ht" off :=
          (inner_all_hits c (innerInput c x) "ht" off).mpr
            (Or.inr ⟨rfl,List.getElem_mem _⟩)
        change mapCell id (final.mem "ht" off)=
          .mk .fp32 ((NativeStores.apply (innerGraphStores c x) emptyState.mem "ht" off).read .fp32)
        rw [cell_identity,memory,NativeStores.at_hit _ _ emptyState.mem _ _ hit]
        exact NativeStores.apply_tag _ _ _ _ _ (inner_all_typed c (innerInput c x)).2 hit
    · intro r off outside
      rw [memory]
      exact NativeStores.frame _ _ r off (inner_outside c x (innerInput c x) r off outside)

theorem prefix_implements (c : HostFP.Cumulative) : (prefixIO c).Implements (prefixGraph c) := by
  apply ProgramIO.Implements.intro
  · intro x bounds bounded s _
    exact prefix_io_safe c x bounds bounded s
  · intro x s bound
    obtain ⟨final,ran,memory⟩ := prefix_run c (allJobs c.plans) s
    have writes : (allJobs c.plans).flatMap (fun j=>prefixStores c j s)=prefixGraphStores c x := by
      simp only [prefixGraphStores,prefix_binding c x s bound]
    rw [writes] at memory
    refine ⟨final,ran,?_,?_⟩
    · intro o j _
      change Fin 1 at o
      fin_cases o
      let off := (collect (prefixCells c))[j.1]
      have hit : NativeStores.Hits (prefixGraphStores c x) "o" off :=
        (prefix_all_hits c (prefixInput c x) "o" off).mpr ⟨rfl,List.getElem_mem _⟩
      change mapCell id (final.mem "o" off)=
        .mk .fp32 ((NativeStores.apply (prefixGraphStores c x) emptyState.mem "o" off).read .fp32)
      rw [cell_identity,memory,NativeStores.at_hit _ _ emptyState.mem _ _ hit]
      exact NativeStores.apply_tag _ _ _ _ _ (prefix_all_typed c (prefixInput c x)) hit
    · intro r off outside
      rw [memory]
      exact NativeStores.frame _ _ r off (prefix_outside c x (prefixInput c x) r off outside)

#axiomsClean inner_implements
#axiomsClean prefix_implements
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
