import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeWindows
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeProgramSafety

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000

def innerStores (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr) :=
  NativeStores.inner (NativeGrid.innerShape c) (plan j).base (atJob j s) c.gateK c.hasSeed c.storeFinal
def prefixStores (c : HostFP.Cumulative) (j : Job c.plans) (s : State NativeGraph.Expr) :=
  NativeStores.prefixStores (NativeGrid.prefixShape c) (plan j).base (atJob j s)

theorem innerStores_same (c : HostFP.Inner) (j : Job c.plans) (s t : State NativeGraph.Expr)
    (inputs : ∀ (r : RegionName) off, r≠"h" → r≠"ht" → s.mem r off=t.mem r off) :
    innerStores c j s=innerStores c j t := by
  apply NativeStores.inner_eq
  refine ⟨rfl,?_,?_,?_,?_,?_,?_,?_⟩
  all_goals intros; congr 1; exact inputs _ _ (by decide) (by decide)

theorem prefixStores_same (c : HostFP.Cumulative) (j : Job c.plans) (s t : State NativeGraph.Expr)
    (inputs : ∀ off, s.mem "s" off=t.mem "s" off) :
    prefixStores c j s=prefixStores c j t := by
  apply NativeStores.prefix_eq
  refine ⟨rfl,?_⟩
  intros; congr 1; exact inputs _

theorem inner_single (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr) :
    ∃ final, (invocation (NativeGrid.innerSource c) j).run s=some final ∧
      final.mem=NativeStores.apply (innerStores c j s) s.mem := by
  obtain ⟨final,ran,memory⟩ := NativeValue.source_memory (NativeGrid.innerShape c)
    (plan j).base c.gateK c.hasSeed c.storeFinal (atJob j s)
  refine ⟨final,ran,?_⟩
  rw [memory,NativeStores.inner_memory]
  rfl

theorem prefix_single (c : HostFP.Cumulative) (j : Job c.plans) (s : State NativeGraph.Expr) :
    ∃ final, (invocation (NativeGrid.prefixSource c) j).run s=some final ∧
      final.mem=NativeStores.apply (prefixStores c j s) s.mem := by
  obtain ⟨final,ran,memory⟩ := NativePrefix.source_memory (NativeGrid.prefixShape c)
    (plan j).base (atJob j s)
  refine ⟨final,ran,?_⟩
  rw [memory,NativeStores.prefix_memory]
  rfl

theorem innerStores_frame (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) (hr : r≠"h") (ht : r≠"ht") :
    NativeStores.apply (innerStores c j s) s.mem r off=s.mem r off := by
  apply NativeStores.frame
  rw [innerStores,NativeStores.inner_hits]
  rintro (⟨_,_,h,_⟩|⟨_,h,_⟩)
  · exact hr h
  · exact ht h

theorem prefixStores_frame (c : HostFP.Cumulative) (j : Job c.plans) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) (hr : r≠"o") :
    NativeStores.apply (prefixStores c j s) s.mem r off=s.mem r off := by
  apply NativeStores.frame
  rw [prefixStores,NativeStores.prefix_hits]
  rintro ⟨_,h,_⟩
  exact hr h

/-- Every invocation executes the actual selected source. Previously written
outputs cannot alter any later native recurrence because all read roles are
disjoint from the two output roles. -/
theorem inner_run (c : HostFP.Inner) (jobs : List (Job c.plans)) (s : State NativeGraph.Expr) :
    ∃ final, runSchedule (jobs.map (invocation (NativeGrid.innerSource c))) s=some final ∧
      final.mem=NativeStores.apply (jobs.flatMap (fun j => innerStores c j s)) s.mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons j rest ih =>
    obtain ⟨mid,ran,memory⟩ := inner_single c j s
    obtain ⟨final,tail,finalMemory⟩ := ih mid
    have same : ∀ k, innerStores c k mid=innerStores c k s := by
      intro k
      apply innerStores_same
      intro r off hr ht
      rw [memory]
      exact innerStores_frame c j s r off hr ht
    refine ⟨final,?_,?_⟩
    · simpa only [List.map_cons,runSchedule,ran,Option.bind_some] using tail
    · rw [finalMemory]
      simp only [same,List.flatMap_cons,NativeStores.apply_append,memory]

theorem prefix_run (c : HostFP.Cumulative) (jobs : List (Job c.plans)) (s : State NativeGraph.Expr) :
    ∃ final, runSchedule (jobs.map (invocation (NativeGrid.prefixSource c))) s=some final ∧
      final.mem=NativeStores.apply (jobs.flatMap (fun j => prefixStores c j s)) s.mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons j rest ih =>
    obtain ⟨mid,ran,memory⟩ := prefix_single c j s
    obtain ⟨final,tail,finalMemory⟩ := ih mid
    have same : ∀ k, prefixStores c k mid=prefixStores c k s := by
      intro k
      apply prefixStores_same
      intro off
      rw [memory]
      exact prefixStores_frame c j s "s" off (by decide)
    refine ⟨final,?_,?_⟩
    · simpa only [List.map_cons,runSchedule,ran,Option.bind_some] using tail
    · rw [finalMemory]
      simp only [same,List.flatMap_cons,NativeStores.apply_append,memory]

theorem schedule_safe {J : Type} (jobs : List J) (launch : J → Invocation NativeGraph.Expr)
    (bounds : RegionName → Nat) (safe : ∀ j ∈ jobs, ∀ s, (launch j).Safe bounds s)
    (s : State NativeGraph.Expr) : ScheduleSafe bounds (jobs.map launch) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons j rest ih =>
    exact ⟨safe j List.mem_cons_self s,
      fun t _ => ih (fun k member => safe k (List.mem_cons_of_mem j member)) t⟩

#axiomsClean inner_run
#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
