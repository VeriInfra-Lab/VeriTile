import bench.tritonbench_g.cache_transform_triton.IO
import bench.tritonbench_g.cache_transform_triton.Proofs.IOSafety
import bench.tritonbench_g.cache_transform_triton.Proofs.KernelInterface
import VeriTile.Triton.Float.InputBindingIO

namespace VeriTile.Bench.Repairs.CacheTransform.IOProof
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

/-- Match the shared fixed schedule runner to the retained source launch. -/
theorem jobs_launch {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (k : Nat) : runSchedule (jobs M p k) s=KernelSource.launch M p s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [jobs,List.range_succ,List.map_append,List.map_singleton,runSchedule_append]
    rw [show runSchedule ((List.range k).map (fun pid =>
      ({kernel:=source p,backend:=M,pids:=fun ax => if ax=0 then pid else 0} : Invocation α))) s =
        KernelSource.launch M p s k from ih]
    cases ran : KernelSource.launch M p s k <;>
      simp [KernelSource.launch,ran,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,KernelSource.positioned]

/-- Exact IO cells instantiate the existing typed table and payload bindings. -/
theorem inputs {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) (domain : CacheIO.Domain M p n capacity xs)
    (initial : State α) (cells : (cacheIO M p n capacity).InputCells xs initial)
    (positive : 0<p.count) :
    GatherInput p initial n capacity (fun q => (CacheIO.row M p n capacity xs q).toNat)
      (CacheIO.payload M p n capacity xs) := by
  obtain ⟨shape,width,cacheDistinct,rowsDistinct,capacityBound,rows⟩ := domain positive
  refine ⟨shape,width,cacheDistinct,rowsDistinct,?_,fun q hq => (rows q hq).2,capacityBound,?_⟩
  · intro q hq
    rw [Int.toNat_of_nonneg (rows q hq).1]
    have pin := cells (0 : Fin 2) (⟨q,hq⟩,PUnit.unit) positive
    change initial.mem p.rows (q*1+0)=Cell.mk .int (xs (0 : Fin 2) (⟨q,hq⟩,PUnit.unit)) at pin
    simpa [CacheIO.row,hq] using pin
  · intro q hq
    have pin := cells (1 : Fin 2) (⟨q,hq⟩,PUnit.unit) positive
    change initial.mem p.cache (q*1+0)=Cell.mk .nat (xs (1 : Fin 2) (⟨q,hq⟩,PUnit.unit)) at pin
    simpa [CacheIO.payload,hq] using pin

private theorem capacity_of_lanes (n cap : Nat) (h : ∀ j : Fin n,j.val<cap) : n≤cap := by
  by_contra hn
  have lt : cap<n := by omega
  have impossible : cap<cap := h ⟨cap,lt⟩
  exact Nat.lt_irrefl _ impossible

/-- The full declared windows imply the three capacities used by the source. -/
theorem capacities {α : Type} (M : Algebra α) (p : Parameters) (n capacity : Nat)
    (xs : (cacheIO M p n capacity).Inputs) (bounds : RegionName → Nat)
    (bounded : (cacheIO M p n capacity).InBounds bounds xs) (positive : 0<p.count) :
    n≤bounds p.rows ∧ capacity*p.rowBytes≤bounds p.cache ∧ p.count≤bounds p.output := by
  refine ⟨capacity_of_lanes _ _ ?_,capacity_of_lanes _ _ ?_,capacity_of_lanes _ _ ?_⟩
  · intro j
    have h := bounded.2.1 (0 : Fin 2) (j,PUnit.unit) positive
    change j.val*1+0<bounds p.rows at h
    simpa using h
  · intro j
    have h := bounded.2.1 (1 : Fin 2) (j,PUnit.unit) positive
    change j.val*1+0<bounds p.cache at h
    simpa using h
  · intro j
    have h := bounded.2.2.1 (0 : Fin 1) (j,PUnit.unit) trivial
    change j.val*1+0<bounds p.output at h
    simpa using h

/-- Every successive source runs on the actual predecessor memory. -/
theorem safe_grid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (input : 0<p.count → GatherInput p s n capacity rows cache)
    (bounds : RegionName → Nat)
    (caps : 0<p.count → n≤bounds p.rows ∧ capacity*p.rowBytes≤bounds p.cache ∧ p.count≤bounds p.output)
    (k : Nat) (inside : k≤blocks p.count) : ScheduleSafe bounds (jobs M p k) s := by
  induction k with
  | zero => simp [jobs,ScheduleSafe]
  | succ k ih =>
    have positive : 0<p.count := lt_of_le_of_lt (Nat.zero_le _) (block_base_bound p.count k (by omega))
    have prior := ih (by omega)
    simp only [jobs,List.range_succ,List.map_append,List.map_singleton,scheduleSafe_append]
    refine ⟨prior,?_⟩
    intro t ran
    have run := KernelInterface.launch_success M p start block grid s k (by omega)
    rw [←jobs_launch] at run
    have same : t=runTiles M p s k := Option.some.inj (ran.symm.trans run)
    subst t
    simp only [scheduleSafe_singleton]
    let job : Invocation α := {kernel:=source p,backend:=M,pids:=fun ax => if ax=0 then k else 0}
    have h := run_input M p s n capacity rows cache (input positive) k
    have hi : GatherInput p (job.initial (runTiles M p s k)) n capacity rows cache := by
      refine ⟨h.shape,h.width,h.distinctCache,h.distinctRows,?_,h.rowBound,h.capacityBound,?_⟩
      · intro q hq
        simpa only [Invocation.initial_mem] using h.table q hq
      · intro q hq
        simpa only [Invocation.initial_mem] using h.payload q hq
    exact access_safe_of_input M p (job.initial (runTiles M p s k)) n capacity rows cache
      hi positive bounds (caps positive).1 (caps positive).2.1 (caps positive).2.2

end
end VeriTile.Bench.Repairs.CacheTransform.IOProof
