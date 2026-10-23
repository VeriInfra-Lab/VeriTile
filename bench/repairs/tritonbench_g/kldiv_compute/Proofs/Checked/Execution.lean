import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.Pointers

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous.Native
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite (bytes SpanDisjoint)
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Source
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

variable {M A : Type} (ops : MemoryPrimitives M A)

def read (L : Layout n) (s : M) (r : Slot) (i : Nat) : A :=
  ops.read s .fp32 (address (L.window r) i)

def writes (a : DataView) (values : Nat → A) (indices : List Nat) : List (Write A) :=
  indices.map fun i => ⟨address a i,values i⟩

def result (compute : A → A → A) (L : Layout n) (s : M) (i : Nat) : A :=
  compute (read ops L s .x i) (read ops L s .y i)

/-- Exact original finite source, whose AST lowering and every instruction's
execution were already proved in the unchanged helper. The launch uses the checked extent and original tensor pointers. -/
def runJob (compute : A → A → A) (undef : A) (L : Layout n) (job : Nat) (s : M) : Option M :=
  Source.runProgram ops compute (views L 0) undef "x" "y" "out"
    (n) job s

def runJobs (compute : A → A → A) (undef : A) (L : Layout n) : List Nat → M → Option M
  | [],s => some s
  | job::rest,s => do
      let next ← runJob ops compute undef L job s
      runJobs compute undef L rest next

theorem source_job (compute : A → A → A) (undef : A) (L : Layout n)
    (job : Nat) (valid : Index.ValidJob n job) (s : M) :
    runJob ops compute undef L job s =
      some (applyWrites ops .fp32 (writes (L.window .out) (result ops compute L s) (Index.indices n job)) s) := by
  rw [runJob,Source.source_program_reduction]
  have vx : views L 0 "x"=view (L.window .x) 0 := views_slot L _ .x
  have vy : views L 0 "y"=view (L.window .y) 0 := views_slot L _ .y
  have vo : views L 0 "out"=view (L.window .out) 0 := views_slot L _ .out
  simp only [vx,vy,vo]
  congr 2
  unfold Source.uncheckedWrites Index.indices writes
  simp only [List.map_map]
  apply List.map_congr_left
  intro lane hl
  simp only [Function.comp_def,lane_exact L job valid .x lane hl,
    lane_exact L job valid .y lane hl,lane_exact L job valid .out lane hl,result,read]
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

theorem result_preserved (compute : A → A → A) (L : Layout n) (s t : M)
    (frame : Preserved ops (view (L.window .out) 0) t s) (i : Nat) (hi : i<n) :
    result ops compute L t i = result ops compute L s i := by
  have hx := input_preserved ops (view (L.window .x) 0) (view (L.window .out) 0)
    t s frame (output_disjoint_x L 0) _ (inside L .x i hi 0)
  have hy := input_preserved ops (view (L.window .y) 0) (view (L.window .out) 0)
    t s frame (output_disjoint_y L 0) _ (inside L .y i hi 0)
  exact congrArg₂ compute hx hy

/-- Programs can arrive in any order and even repeat: fresh input snapshots
make each overlapping repeated write identical. Completeness is added below. -/
theorem jobs_correct (compute : A → A → A) (undef : A) (L : Layout n)
    (memAPI : LocalMemory ops L) (js : List Nat)
    (valid : ∀ j ∈ js,Index.ValidJob n j) (s : M) :
    ∃ t,runJobs ops compute undef L js s=some t ∧
      (∀ i,i<n → read ops L t .out i =
        if ∃ j ∈ js,i ∈ Index.indices n j then result ops compute L s i else read ops L s .out i) ∧
      Preserved ops (view (L.window .out) 0) t s := by
  induction js generalizing s with
  | nil => exact ⟨s,rfl,by simp, preserved_refl _ _ _⟩
  | cons j rest ih =>
    let mid := applyWrites ops .fp32
      (writes (L.window .out) (result ops compute L s) (Index.indices n j)) s
    have hmid := source_job ops compute undef L j (valid j (by simp)) s
    have fm := job_frame ops L (result ops compute L s) j (valid j (by simp)) s
    obtain ⟨t,ht,hv,ft⟩ := ih (fun k hk => valid k (by simp [hk])) mid
    refine ⟨t,by simp only [runJobs,hmid]; exact ht,?_,preserved_trans _ _ _ _ _ ft fm⟩
    intro i hi
    rw [hv i hi,result_preserved ops compute L s mid fm i hi]
    have hm := writes_read ops L memAPI (result ops compute L s) (Index.indices n j) s i
    change read ops L mid .out i = _ at hm
    rw [hm]
    by_cases hr : ∃ k ∈ rest,i ∈ Index.indices n k
    · simp [hr]
    · by_cases hj : i ∈ Index.indices n j
      · simp [hr,hj]
      · simp [hr,hj]

structure LaunchResult (compute : A → A → A) (undef : A) (L : Layout n)
    (schedule : Index.Schedule n) (s t : M) : Prop where
  execution : runJobs ops compute undef L (Index.jobs n schedule) s=some t
  values : ∀ i,i<n → read ops L t .out i=result ops compute L s i
  frame : Preserved ops (view (L.window .out) 0) t s
  sourceAddresses : ∀ job ∈ Index.jobs n schedule,∀ lane,
    lane ∈ Source.programLanes n job →
    (∀ r,(view (L.window r) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane) =
      address (L.window r) (job*1024+lane)) ∧
    (∀ r,ReadOK ops (view (L.window r) 0)
      ((view (L.window r) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane))) ∧
    WriteOK ops (view (L.window .out) 0)
      ((view (L.window .out) 0).nativeAt (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 job lane))

theorem launch_correct (compute : A → A → A) (undef : A) (L : Layout n)
    (memAPI : LocalMemory ops L) (schedule : Index.Schedule n) (s : M) :
    ∃ t,LaunchResult ops compute undef L schedule s t := by
  obtain ⟨t,he,hv,hf⟩ := jobs_correct ops compute undef L memAPI (Index.jobs n schedule)
    (Index.valid_of_mem n schedule) s
  refine ⟨t,he,?_,hf,?_⟩
  · intro i hi
    rw [hv i hi,if_pos (Index.covered n schedule i hi)]
  · intro job hj lane hl
    have valid := Index.valid_of_mem n schedule job hj
    exact ⟨fun r => pointer_exact L job valid r lane hl,
      fun r => read_ok ops L memAPI job valid r lane hl,
      write_ok ops L memAPI job valid lane hl⟩

#specNonCircular result avoiding [runJob,runJobs,Source.runProgram,Source.executeBody,Source.executeInstruction,applyWrites]
#axiomsClean source_job
#axiomsClean writes_read
#axiomsClean jobs_correct
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Contiguous.Native
