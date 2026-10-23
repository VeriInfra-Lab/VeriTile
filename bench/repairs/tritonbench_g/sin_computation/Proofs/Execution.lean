import bench.repairs.tritonbench_g.sin_computation.Proofs.Pointers
import bench.repairs.tritonbench_g.sin_computation.Proofs.Source

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Native
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite (bytes SpanDisjoint)
open VeriTile.Bench.TritonBenchG.AddExample.Finite

noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

variable {M A : Type} (ops : MemoryPrimitives M A)

def read (L : Layout n) (s : M) (r : Slot) (i : Nat) : A :=
  ops.read s .fp32 (address (L.window r) i)

def writes (a : DataView) (values : Nat → A) (indices : List Nat) : List (Write A) :=
  indices.map fun i => ⟨address a i,values i⟩

def result (compute : A → A) (L : Layout n) (s : M) (i : Nat) : A :=
  compute (read ops L s .x i)

/-- Exact original finite source, whose AST lowering and every instruction's
execution were already proved in the unchanged helper. The launch uses the checked extent and original tensor pointers. -/
def runJob (compute : A → A) (undef : A) (L : Layout n) (job : Nat) (s : M) : Option M :=
  Source.runProgram ops compute (views L 0) undef
    (n) job s

def runJobs (compute : A → A) (undef : A) (L : Layout n) : List Nat → M → Option M
  | [],s => some s
  | job::rest,s => do
      let next ← runJob ops compute undef L job s
      runJobs compute undef L rest next

theorem source_job (compute : A → A) (undef : A) (L : Layout n)
    (job : Nat) (valid : Index.ValidJob n job) (s : M) :
    runJob ops compute undef L job s =
      some (applyWrites ops .fp32 (writes (L.window .out) (result ops compute L s) (Index.indices n job)) s) := by
  rw [runJob,Source.source_program_reduction]
  have vx : views L 0 "x_ptr"=view (L.window .x) 0 := views_slot L _ .x
  have vo : views L 0 "output_ptr"=view (L.window .out) 0 := views_slot L _ .out
  simp only [vx,vo]
  congr 2
  unfold Source.uncheckedWrites Index.indices writes
  simp only [List.map_map]
  apply List.map_congr_left
  intro lane hl
  have le (r : Slot) : Source.laneAddress (view (L.window r) 0) job lane =
      address (L.window r) (job*4+lane) := by
    change (view (L.window r) 0).lane job lane = _
    exact lane_exact L job valid r lane hl
  simp only [Function.comp_def,le,result,read]
  rfl

theorem address_disjoint (a : DataView) (i j : Nat) (h : i ≠ j) :
    SpanDisjoint (address a i) (address a i+bytes .fp32) (address a j) (address a j+bytes .fp32) := by
  simp only [SpanDisjoint,address,bytes]
  omega

theorem writes_read (L : Layout n) (memAPI : LocalMemory ops L) (values : Nat → A)
    (is : List Nat) (s : M) (i : Nat) :
    read ops L (applyWrites ops .fp32 (writes (L.window .out) values is) s) .out i =
      if i ∈ is then values i else read ops L s .out i := by
  induction is generalizing s with
  | nil => simp [writes,applyWrites]
  | cons j rest ih =>
    simp only [writes,List.map_cons,applyWrites]
    rw [←writes,ih]
    by_cases hr : i ∈ rest
    · simp [hr]
    · by_cases he : i=j
      · subst j
        simp only [List.mem_cons,hr,or_false,if_false,read]
        exact memAPI.read_write s _ _
      · simp only [List.mem_cons,hr,he,or_false,if_false,read]
        exact ops.read_disjoint s .fp32 (address (L.window .out) j) (values j)
          .fp32 (address (L.window .out) i) (address_disjoint _ j i (Ne.symm he))

theorem job_frame (L : Layout n) (values : Nat → A) (job : Nat)
    (valid : Index.ValidJob n job) (s : M) :
    Preserved ops (view (L.window .out) 0)
      (applyWrites ops .fp32 (writes (L.window .out) values (Index.indices n job)) s) s := by
  apply writes_preserved ops (view (L.window .out) 0)
  intro w hw
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp hw
  exact inside L .out i (Index.indices_bound n job valid i hi) 0

/-- Off-diagonal writes preserve input reads, including exact in-place use. -/
theorem writes_input_untouched (L : Layout n) (values : Nat → A)
    (is : List Nat) (active : ∀ j∈is,j<n) (s : M) (i : Nat) (hi : i<n)
    (outside : i∉is) :
    read ops L (applyWrites ops .fp32 (writes (L.window .out) values is) s) .x i =
      read ops L s .x i := by
  induction is generalizing s with
  | nil => rfl
  | cons j rest ih =>
    have hj : j<n := active j (by simp)
    have hne : i ≠ j := by intro eq; apply outside; simp [eq]
    have hr : i∉rest := fun h => outside (by simp [h])
    simp only [writes,List.map_cons,applyWrites]
    rw [←writes,ih (fun k hk => active k (by simp [hk])) _ hr]
    exact ops.read_disjoint s .fp32 (address (L.window .out) j) (values j)
      .fp32 (address (L.window .x) i) (cross_cells L i j hi hj hne)

theorem index_owner (n pid i : Nat) (hi : i∈Index.indices n pid) : i/4=pid := by
  obtain ⟨lane,hl,rfl⟩ := List.mem_map.mp hi
  have bound := (List.mem_filter.mp hl).1
  have hlt := List.mem_range.mp bound
  omega

/-- Every program executes once. This retains exact in-place correctness;
shifted overlap is excluded by the checked compatible-span condition. -/
theorem jobs_correct (compute : A → A) (undef : A) (L : Layout n)
    (memAPI : LocalMemory ops L) (js : List Nat)
    (valid : ∀ j∈js,Index.ValidJob n j) (unique : js.Nodup) (s : M) :
    ∃ t,runJobs ops compute undef L js s=some t ∧
      (∀ i,i<n → read ops L t .out i =
        if ∃ j∈js,i∈Index.indices n j then result ops compute L s i else read ops L s .out i) ∧
      Preserved ops (view (L.window .out) 0) t s := by
  induction js generalizing s with
  | nil => exact ⟨s,rfl,by simp,preserved_refl _ _ _⟩
  | cons j rest ih =>
    have uj := (List.nodup_cons.mp unique).1
    have ur := (List.nodup_cons.mp unique).2
    let mid := applyWrites ops .fp32
      (writes (L.window .out) (result ops compute L s) (Index.indices n j)) s
    have hmid := source_job ops compute undef L j (valid j (by simp)) s
    have fm := job_frame ops L (result ops compute L s) j (valid j (by simp)) s
    obtain ⟨t,ht,hv,ft⟩ := ih (fun k hk => valid k (by simp [hk])) ur mid
    refine ⟨t,by simp only [runJobs,hmid]; exact ht,?_,preserved_trans _ _ _ _ _ ft fm⟩
    intro i hi
    rw [hv i hi]
    have hm := writes_read ops L memAPI (result ops compute L s) (Index.indices n j) s i
    change read ops L mid .out i = _ at hm
    by_cases hr : ∃ k∈rest,i∈Index.indices n k
    · have outside : i∉Index.indices n j := by
        intro hj
        obtain ⟨k,hk,hki⟩ := hr
        have eq : j=k := (index_owner n j i hj).symm.trans (index_owner n k i hki)
        exact uj (eq ▸ hk)
      have keep := writes_input_untouched ops L (result ops compute L s)
        (Index.indices n j) (fun k hk => Index.indices_bound n j (valid j (by simp)) k hk)
        s i hi outside
      change read ops L mid .x i = read ops L s .x i at keep
      simp only [if_pos hr,result,keep]
      simp [hr,result]
    · simp only [if_neg hr]
      rw [hm]
      by_cases hj : i∈Index.indices n j <;> simp [hr,hj]

structure LaunchResult (compute : A → A) (undef : A) (L : Layout n)
    (schedule : Index.Schedule n) (s t : M) : Prop where
  execution : runJobs ops compute undef L schedule.programs s=some t
  values : ∀ i,i<n → read ops L t .out i=result ops compute L s i
  frame : Preserved ops (view (L.window .out) 0) t s
  sourceAddresses : ∀ job∈schedule.programs,∀ lane,lane∈programLanes n job →
    (∀ r,(view (L.window r) 0).nativeLane job lane=address (L.window r) (job*4+lane)) ∧
    (∀ r,ReadOK ops (view (L.window r) 0) ((view (L.window r) 0).nativeLane job lane)) ∧
    WriteOK ops (view (L.window .out) 0) ((view (L.window .out) 0).nativeLane job lane)

theorem launch_correct (compute : A → A) (undef : A) (L : Layout n)
    (memAPI : LocalMemory ops L) (schedule : Index.Schedule n) (s : M) :
    ∃ t,LaunchResult ops compute undef L schedule s t := by
  obtain ⟨t,he,hv,hf⟩ := jobs_correct ops compute undef L memAPI schedule.programs
    (Index.valid_of_mem n schedule) (schedule.complete.nodup_iff.mpr List.nodup_range) s
  refine ⟨t,he,?_,hf,?_⟩
  · intro i hi
    have cover : ∃ j∈schedule.programs,i∈Index.indices n j := Index.covered n schedule i hi
    rw [hv i hi,if_pos cover]
  · intro job hj lane hl
    have valid := Index.valid_of_mem n schedule job hj
    exact ⟨fun r => pointer_exact L job valid r lane hl,
      fun r => read_ok ops L memAPI job valid r lane hl,
      write_ok ops L memAPI job valid lane hl⟩

#axiomsClean source_job
#axiomsClean jobs_correct
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Native
