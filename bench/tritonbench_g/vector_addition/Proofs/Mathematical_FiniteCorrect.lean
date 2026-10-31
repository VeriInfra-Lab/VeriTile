import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteRoutes
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteSchedule
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FiniteIndices

/-! Finite source acceptance contract for the original `add` wrapper and its
unchanged 1024-lane kernel. Trusted Torch metadata/allocation and ordinary typed
memory primitives are explicit. Strides are metadata: source pointers advance
linearly from data_ptr and storage_offset, including for noncontiguous inputs.
The safe/defined domain is applicability, never an invented wrapper check. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.Finite
open VeriTile.Triton DTypeScope TypedHost
open HostWrapper (Tensor Result Run Event Failure parameters successTrace cudaGuard add)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
local instance : Inhabited TypedHost.Word := ⟨FP.Value.zero .fp32⟩

structure LaunchSafety (x y output : View) (n : Nat) : Prop where
  alignment : ∀ v∈[x,y,output], ∀ i, v.address i % bytes v.dtype=0
  indices : ∀ pid lane, pid<(n+1023)/1024 → lane<1024 →
    (start32 pid).toInt=(pid*1024 : Nat) ∧
    (offset32 pid lane).toInt=(pid*1024+lane : Nat) ∧
    mask32 n pid lane=decide (pid*1024+lane<n)
  pointers : ∀ v∈[x,y,output], ∀ pid lane, pid<(n+1023)/1024 → lane<1024 →
    (pointer64 v pid lane).toNat=v.address (pid*1024+lane)
  accesses : ∀ v∈[x,y,output], ∀ i byte, i<n → byte<bytes v.dtype →
    v.storageBase≤v.address i+byte ∧ v.address i+byte<v.storageBase+v.storageBytes ∧ v.address i+byte<2^64
  ownership : ∀ i, i<n → i/1024<(n+1023)/1024 ∧ i%1024<1024 ∧ (i/1024)*1024+i%1024=i
  outputDistinct : ∀ i j, i≠j → SpanDisjoint (output.address i) (output.finish i) (output.address j) (output.finish j)

theorem launch_safety (x y output : View) (n : Nat) (hn : n<2^31)
    (hx : x.Valid n (rounded n)) (hy : y.Valid n (rounded n)) (ho : output.Valid n (rounded n))
    (ax : x.Aligned) (ay : y.Aligned) (ao : output.Aligned) :
    LaunchSafety x y output n := by
  have valid : ∀ v∈[x,y,output], v.Valid n (rounded n) := by
    intro v hv
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hv
    rcases hv with rfl|rfl|rfl
    · exact hx
    · exact hy
    · exact ho
  have align : ∀ v∈[x,y,output], v.Aligned := by
    intro v hv
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hv
    rcases hv with rfl|rfl|rfl
    · exact ax
    · exact ay
    · exact ao
  exact ⟨fun v hv i ↦ address_aligned v (align v hv) i,
    fun pid lane hp hl ↦ source_index_arithmetic n pid lane hn hp hl,
    fun v hv pid lane hp hl ↦ source_pointer_exact v n pid lane hn (valid v hv) hp hl,
    fun v hv i byte hi hb ↦ active_byte_access v n i byte (valid v hv) hi hb,
    fun i hi ↦ active_owner n i hi,fun i j hij ↦ distinct_cells output i j hij⟩

def physicalRun {M : Type} (b : Numerics) (r : Route) (ops : MemoryPrimitives M (Carrier r))
    (x y output : View) (n : Nat) (physical : M) : M :=
  runBatches ops (scalarSource b r) x y output (sourceBatches n) physical

structure PhysicalCorrect {M σ : Type} (b : Numerics) (r : Route)
    (ops : MemoryPrimitives M (Carrier r)) (p : ValueProofs.Parameters)
    (allocated final : World σ) (physical : M) (x y output : View) : Prop where
  values : ∀ i, i<p.n → readView ops (physicalRun b r ops x y output p.n physical) output i=
    sourceCell b r p allocated i
  logicalResult : ∀ i, i<p.n → readView ops (physicalRun b r ops x y output p.n physical) output i=
    logicalRead r final p.output i
  inputs : InputsPreserved ops x y p.n physical (physicalRun b r ops x y output p.n physical)
  schedules : ∀ jobs, (∀ i, i∈jobs ↔ i<p.n) → ∀ i, i<p.n →
    readView ops (runLive ops (scalarSource b r) x y output jobs physical) output i=sourceCell b r p allocated i
  frame : ∀ address, ¬ output.active p.n address →
    ops.observeByte (physicalRun b r ops x y output p.n physical) address=ops.observeByte physical address

theorem physical_correct {M σ : Type} (b : Numerics) (r : Route)
    (ops : MemoryPrimitives M (Carrier r)) (p : ValueProofs.Parameters)
    (allocated final : World σ) (physical : M) (x y output : View)
    (hx : x.Valid p.n (rounded p.n)) (hy : y.Valid p.n (rounded p.n)) (ho : output.Valid p.n (rounded p.n))
    (dx : StorageDisjoint output x) (dy : StorageDisjoint output y)
    (inputs : InputsRelation r ops p allocated physical x y)
    (formula : project r final=expected b r p allocated) :
    PhysicalCorrect b r ops p allocated final physical x y output := by
  have hc : physicalRun b r ops x y output p.n physical=
      runLive ops (scalarSource b r) x y output (activeLanes p.n) physical :=
    source_batches_correspondence ops (scalarSource b r) x y output p.n physical hx hy ho dx dy
  have result := physical_output_matches b r ops p allocated final physical x y output hx hy ho dx dy inputs formula
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro i hi
    rw [hc,result i hi]
    exact output_cell_of_project b r p allocated final formula i hi
  · intro i hi
    rw [hc]
    exact result i hi
  · rw [hc]
    exact live_preserves_inputs ops (scalarSource b r) x y output p.n (activeLanes p.n) physical physical
      (fun i hi ↦ (mem_activeLanes p.n i).1 hi) hx hy ho dx dy (fun _ _ ↦ ⟨rfl,rfl⟩)
  · intro jobs complete
    exact physical_schedule_output b r ops p allocated physical x y output jobs hx hy ho dx dy inputs complete
  · intro address outside
    rw [hc]
    exact live_byte_frame ops (scalarSource b r) x y output p.n (activeLanes p.n) physical
      (fun i hi ↦ (mem_activeLanes p.n i).1 hi) address outside

#axiomsClean launch_safety
#axiomsClean physical_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.Finite
