import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.NativeIO

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000

theorem address_bound (addresses : List Nat) (capacity : Nat)
    (bound : ∀ i : Fin addresses.length,addresses[i]<capacity)
    (off : Nat) (member : off∈addresses) : off<capacity := by
  obtain ⟨i,hi,rfl⟩ := List.getElem_of_mem member
  exact bound ⟨i,hi⟩

/-- The IO capacity premise proves every actually evaluated load/store bound,
including endpoint gates and all iterations of the recurrent source. -/
theorem inner_job_safe (c : HostFP.Inner) (x : (innerIO c).Inputs) (bounds : RegionName → Nat)
    (bounded : (innerIO c).InBounds bounds x) (j : Job c.plans) (s : State NativeGraph.Expr) :
    (invocation (NativeGrid.innerSource c) j).Safe bounds s := by
  apply NativeSafety.inner_safe
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro hs i
    exact address_bound _ _ (fun k=>bounded.2.1 (3:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (seed_mem c j s hs i))
  · intro n hn i
    exact address_bound _ _ (fun k=>bounded.2.1 (0:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (key_mem c j s n hn i))
  · intro n hn i
    exact address_bound _ _ (fun k=>bounded.2.1 (1:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (value_mem c j s n hn i))
  · intro hg n hn i
    exact address_bound _ _ (fun k=>bounded.2.1 (2:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (gate_key_mem c j s hg n hn i))
  · intro hg n hn i
    exact address_bound _ _ (fun k=>bounded.2.1 (2:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (gate_value_mem c j s hg n hn i))
  · intro hg n hn i
    exact address_bound _ _ (fun k=>bounded.2.1 (2:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (key_end_mem c j s hg n hn i))
  · intro hg n hn i
    exact address_bound _ _ (fun k=>bounded.2.1 (2:Fin 4) (k,PUnit.unit) trivial) _
      (mem_collect _ j (value_end_mem c j s hg n hn i))
  · intro n hn i
    exact address_bound _ _ (fun k=>bounded.2.2.1 (0:Fin 2) (k,PUnit.unit) trivial) _
      (mem_collect _ j (history_mem c j s n hn i))
  · intro hf i
    exact address_bound _ _ (fun k=>bounded.2.2.1 (1:Fin 2) (k,PUnit.unit) trivial) _
      (mem_collect _ j (final_mem c j s hf i))

theorem prefix_job_safe (c : HostFP.Cumulative) (x : (prefixIO c).Inputs) (bounds : RegionName → Nat)
    (bounded : (prefixIO c).InBounds bounds x) (j : Job c.plans) (s : State NativeGraph.Expr) :
    (invocation (NativeGrid.prefixSource c) j).Safe bounds s := by
  apply NativeSafety.prefix_safe
  constructor
  · intro active i
    exact address_bound _ _ (fun k=>bounded.2.1 (0:Fin 1) (k,PUnit.unit) trivial) _
      (mem_collect _ j (prefix_mem c j s active i))
  · intro active i
    exact address_bound _ _ (fun k=>bounded.2.2.1 (0:Fin 1) (k,PUnit.unit) trivial) _
      (mem_collect _ j (prefix_mem c j s active i))

theorem inner_io_safe (c : HostFP.Inner) (x : (innerIO c).Inputs) (bounds : RegionName → Nat)
    (bounded : (innerIO c).InBounds bounds x) (s : State NativeGraph.Expr) :
    ScheduleSafe bounds ((innerIO c).schedule x) s :=
  schedule_safe _ _ bounds (fun j _=>inner_job_safe c x bounds bounded j) s

theorem prefix_io_safe (c : HostFP.Cumulative) (x : (prefixIO c).Inputs) (bounds : RegionName → Nat)
    (bounded : (prefixIO c).InBounds bounds x) (s : State NativeGraph.Expr) :
    ScheduleSafe bounds ((prefixIO c).schedule x) s :=
  schedule_safe _ _ bounds (fun j _=>prefix_job_safe c x bounds bounded j) s

#axiomsClean inner_io_safe
#axiomsClean prefix_io_safe
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
