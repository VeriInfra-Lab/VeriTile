import bench.tritonbench_g.triton_argmax.Proofs.BoundGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.BoundSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Memory
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

def dimEntry {M} (mem : Primitives M) (p : DimMemory.Buffers) (before : M)
    (template : State Word) (program : Nat) : State Word :=
  {template with
    pids:=fun ax => if ax=0 then program else template.pids ax
    mem:=fun region offset =>
      if region=X then .mk .real (mem.read 4 before (address p.input 4 offset))
      else if region=Region.cast O then .mk .int (Int.ofNat (mem.read 8 before (address p.output 8 offset)).toNat)
      else template.mem region offset}

@[simp] theorem dim_entry_input {M} (mem : Primitives M) (p : DimMemory.Buffers) (before : M)
    (template : State Word) (program offset : Nat) :
    ((dimEntry mem p before template program).mem X offset).read .real=DimMemory.input mem p before offset := by
  simp [dimEntry,DimMemory.input]

def groups (c : DimConfig) : List (Nat × Nat) :=
  (List.range ((c.outer+127)/128)).flatMap fun pid => (List.range c.inner).map fun inner => (pid,inner)

def rows (job : Nat × Nat) : List DimMemory.Job :=
  (List.range 128).map fun lane => (job.1*128+lane,job.2)

theorem groups_rows (c : DimConfig) : (groups c).flatMap rows=DimMemory.jobs c := by
  simp [groups,rows,DimMemory.jobs,List.flatMap_assoc,List.flatMap_map]

theorem group_inner (c : DimConfig) (job : Nat × Nat) (member : job∈groups c) : job.2<c.inner := by
  simp only [groups,List.mem_flatMap,List.mem_map,List.mem_range] at member
  obtain ⟨pid,_,inner,hi,same⟩ := member
  cases same
  exact hi

/-- Commit all active output stores of this complete physical CTA. The
payloads are read from the quotation's resulting heap, then encoded as i64. -/
def dimCommit {M} (mem : Primitives M) (c : DimConfig) (p : DimMemory.Buffers)
    (job : Nat × Nat) (quoted : State Word) (before : M) : M :=
  (rows job).foldl (fun state j => if j.1<c.outer then mem.write 8 state
    (address p.output 8 (dimOutputOffset c j.1 j.2))
    (BitVec.ofInt 64 ((quoted.mem (Region.cast O) (dimOutputOffset c j.1 j.2)).read .int)) else state) before

def DimStep {M} (mem : Primitives M) (rest : Algebra Word) (template : State Word)
    (c : DimConfig) (p : DimMemory.Buffers) (index64 : Bool) (job : Nat × Nat) (before after : M) : Prop :=
  ∃ quoted, FP.Structural.exec (Typed.backend rest)
      (JITCall.dimension X O c.outer c.extent c.inner 128 128 0 index64)
      (dimEntry mem p before template (job.1*c.inner+job.2))=some quoted ∧
    after=dimCommit mem c p job quoted before ∧
    (∀ region offset,(region≠Region.cast O ∨ ∀ lane : Fin 128,job.1*128+lane.val<c.outer →
      offset≠(job.1*128+lane.val)*c.inner+job.2) →
      quoted.mem region offset=(dimEntry mem p before template (job.1*c.inner+job.2)).mem region offset)

theorem dim_step {M} (mem : Primitives M) (rest : Algebra Word) (template : State Word)
    (c : DimConfig) (g : Native.DimPrograms c) (p : DimMemory.Buffers) (layout : DimMemory.Layout c p)
    (nonempty : 0<c.extent) (index64 : Bool) (job : Nat × Nat) (hi : job.2<c.inner) (before : M) :
    DimStep mem rest template c p index64 job before
      ((rows job).foldl (DimMemory.liveRow mem fp32Scalars c p) before) := by
  have positive : 0<c.inner := by omega
  have quotient : (job.1*c.inner+job.2)/c.inner=job.1 := by
    simpa [Nat.add_comm,Nat.mul_comm,Nat.div_eq_of_lt hi] using Nat.add_mul_div_left job.2 job.1 positive
  have remainder : (job.1*c.inner+job.2)%c.inner=job.2 := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hi]
  obtain ⟨quoted,run,values,frame⟩ := TypedNative.dimension rest
    (dimEntry mem p before template (job.1*c.inner+job.2)) X O c g 0 index64 nonempty positive
  simp only [dimEntry,ite_true,Nat.add_zero,quotient,remainder,dim_entry_input] at values frame
  refine ⟨quoted,run,?_,frame⟩
  have snapshot := DimMemory.grid_snapshot mem fp32Scalars c p layout (rows job)
    (by intro j hj; obtain ⟨lane,_,rfl⟩ := List.mem_map.mp hj; exact hi)
    (DimMemory.input mem p before) before (fun _ _ => rfl)
  rw [snapshot.1]
  have fold : ∀ work : List DimMemory.Job,(∀ j,j∈work → j∈rows job) → ∀ state,
      work.foldl (DimMemory.rowProgram mem fp32Scalars c p (DimMemory.input mem p before)) state=
        work.foldl (fun state j => if j.1<c.outer then mem.write 8 state
          (address p.output 8 (dimOutputOffset c j.1 j.2))
          (BitVec.ofInt 64 ((quoted.mem (Region.cast O) (dimOutputOffset c j.1 j.2)).read .int)) else state) state := by
    intro work valid
    induction work with
    | nil => intro state; rfl
    | cons j work ih =>
      intro state
      rw [List.foldl_cons,List.foldl_cons]
      have one : DimMemory.rowProgram mem fp32Scalars c p (DimMemory.input mem p before) state j=
          (if j.1<c.outer then mem.write 8 state (address p.output 8 (dimOutputOffset c j.1 j.2))
          (BitVec.ofInt 64 ((quoted.mem (Region.cast O) (dimOutputOffset c j.1 j.2)).read .int)) else state) := by
        obtain ⟨lane,hl,rfl⟩ := List.mem_map.mp (valid j (by simp))
        have bound : lane<128 := List.mem_range.mp hl
        by_cases active : job.1*128+lane<c.outer
        · have written := values ⟨lane,bound⟩ active
          simp only [Fin.val_mk] at written
          simp only [DimMemory.rowProgram,DimMemory.key,DimMemory.value,if_pos active,
            dimOutputOffset,Native.dim_index] at *
          rw [written]
          rfl
        · simp [DimMemory.rowProgram,active]
      rw [one]
      exact ih (fun j hj => valid j (by simp [hj])) _
  exact fold _ (fun _ h => h) before

def DimensionRun {M} (mem : Primitives M) (rest : Algebra Word) (template : State Word)
    (c : DimConfig) (p : DimMemory.Buffers) (index64 : Bool) (before after : M) : Prop :=
  Steps (DimStep mem rest template c p index64) (groups c) before after

theorem fold_groups {M α β} (f : M → β → M) (expand : α → List β) (jobs : List α) (s : M) :
    jobs.foldl (fun state job => (expand job).foldl f state) s=(jobs.flatMap expand).foldl f s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job jobs ih => simp [List.foldl_cons,ih,List.foldl_append]

theorem dimension_run {M} (mem : Primitives M) (rest : Algebra Word) (template : State Word)
    (c : DimConfig) (g : Native.DimPrograms c) (p : DimMemory.Buffers) (layout : DimMemory.Layout c p)
    (nonempty : 0<c.extent) (index64 : Bool) (before : M) :
    DimensionRun mem rest template c p index64 before (Native.dimension mem g p before) := by
  have trace := steps_fold (DimStep mem rest template c p index64)
    (fun state job => (rows job).foldl (DimMemory.liveRow mem fp32Scalars c p) state) (groups c) before
    (fun state job member => dim_step mem rest template c g p layout nonempty index64 job (group_inner c job member) state)
  rw [fold_groups,groups_rows] at trace
  rw [Native.dimension_source]
  exact trace

#axiomsClean dim_step
#axiomsClean dimension_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.BoundSource
