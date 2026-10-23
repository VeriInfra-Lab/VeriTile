import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedResult
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.ProgramCorrespondence
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.ProgramOrder

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedGrid
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def enter (s : Shape) (pid : Program) (u : State α) : State α :=
  { u with
    pids := fun ax => if ax=0 then pid.1 else if ax=1 then pid.2.1 else pid.2.2
    numPids := fun ax => if ax=0 then s.rowBlocks else if ax=1 then s.batches else s.heads }

def execute (p : Ops α) (s : Shape) (cumulative64 offsets64 : Bool) :
    List Program → State α → Option (State α)
  | [],u => some u
  | pid::rest,u => do
      let next ← FP.Structural.exec (KernelAlgebra.backend p)
        (TypedExecution.kernel s cumulative64 offsets64) (enter s pid u)
      execute p s cumulative64 offsets64 rest next

def run (p : Ops α) (c : Config) (cumulative64 offsets64 : Bool) (u : State α) : Option (State α) :=
  if c.shape.total=0 ∨ c.shape.half=0 then some u
  else execute p c.shape cumulative64 offsets64 (programs c) u

def runOrder (p : Ops α) (c : Config) (cumulative64 offsets64 : Bool)
    (order : List Program) (u : State α) : Option (State α) :=
  if c.shape.total=0 ∨ c.shape.half=0 then some u
  else execute p c.shape cumulative64 offsets64 order u

def Outside (c : Config) (r : RegionName) (o : Nat) : Prop :=
  ∀ w,c.Active w → ¬(r=KernelAlgebra.regionName c.shape.output ∧
    (o=c.cell w Bool.false ∨ o=c.cell w Bool.true))

theorem entered_metadata (c : Config) (u : State α) (md : ProgramModel.MetadataBinding c u)
    (pid : Program) : ProgramModel.MetadataBinding c (enter c.shape pid u) :=
  ⟨md.cumulative,md.offsets⟩

theorem metadata_after_result (c : Config) (p : Ops α) (u : State α)
    (md : ProgramModel.MetadataBinding c u) :
    ProgramModel.MetadataBinding c (ProgramModel.result c.shape p u) := by
  have cu (i : Nat) := ProgramModel.result_other c.shape p u ("cu" : RegionName) i (by
    cases hi : c.shape.inplace <;> simp [Shape.output,KernelAlgebra.regionName,hi])
  have offsets (i : Nat) := ProgramModel.result_other c.shape p u ("offsets" : RegionName) i (by
    cases hi : c.shape.inplace <;> simp [Shape.output,KernelAlgebra.regionName,hi])
  constructor
  · intro packed i hi
    rw [cu]
    exact md.cumulative packed i hi
  · intro tensor i hi
    rw [offsets]
    exact md.offsets tensor i hi

theorem metadata_mem_eq (c : Config) (u v : State α) (eq : u.mem=v.mem)
    (md : ProgramModel.MetadataBinding c v) : ProgramModel.MetadataBinding c u := by
  constructor
  · intro packed i hi;rw [eq];exact md.cumulative packed i hi
  · intro tensor i hi;rw [eq];exact md.offsets tensor i hi

theorem program_frame (c : Config) (valid : c.Valid) (p : Ops α) (u : State α)
    (md : ProgramModel.MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) (r : RegionName) (o : Nat) (outside : Outside c r o) :
    (ProgramModel.result c.shape p u).mem r o=u.mem r o := by
  by_cases region : r=KernelAlgebra.regionName c.shape.output
  · subst r
    apply ProgramModel.result_frame c valid p u md batch head c.shape.output o
    intro w hw written
    exact outside w (ProgramModel.program_active c u w hw) ⟨rfl,written.2⟩
  · exact ProgramModel.result_other c.shape p u r o region

/-- Every actual source program succeeds. Program memory effects compose
through the existing noninterference and complete schedule model. -/
theorem complete_list (c : Config) (valid : c.Valid) (p : Ops α) (cumulative64 offsets64 : Bool)
    (ps : List Program) (inside : ∀ pid∈ps,pid∈programs c) (initial : State α)
    (md : ProgramModel.MetadataBinding c initial) :
    ∃ final,execute p c.shape cumulative64 offsets64 ps initial=some final ∧
      KernelAlgebra.memory final=sourcePrograms c p ps (KernelAlgebra.memory initial) ∧
      (∀ r o,Outside c r o → final.mem r o=initial.mem r o) := by
  induction ps generalizing initial with
  | nil => exact ⟨initial,rfl,rfl,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    have hp := inside pid (by simp)
    have batch : pid.2.1<c.shape.batches := by
      exact List.mem_range.mp (List.mem_product.mp (List.mem_product.mp hp).2).1
    have head : pid.2.2<c.shape.heads := by
      exact List.mem_range.mp (List.mem_product.mp (List.mem_product.mp hp).2).2
    let entry := enter c.shape pid initial
    have boundB : entry.pids 1<c.shape.batches := by simpa [entry,enter] using batch
    have boundH : entry.pids 2<c.shape.heads := by simpa [entry,enter] using head
    have inputMD : ProgramModel.MetadataBinding c entry := entered_metadata c initial md pid
    obtain ⟨next,executed⟩ := TypedExecution.executes (KernelAlgebra.backend p) c.shape cumulative64 offsets64 entry
    have mem : next.mem=(ProgramModel.result c.shape p entry).mem := by
      have got := TypedExecution.memory_result p c.shape cumulative64 offsets64 entry
      simpa only [executed,Option.map_some,Option.some.injEq] using got
    have nextMD := metadata_mem_eq c next _ mem (metadata_after_result c p entry inputMD)
    have nextMemory : KernelAlgebra.memory next=sourceProgram c p pid (KernelAlgebra.memory initial) := by
      have model := ProgramModel.source_program c valid p entry inputMD boundB boundH
      have nextEq : KernelAlgebra.memory next=KernelAlgebra.memory (ProgramModel.result c.shape p entry) := by
        unfold KernelAlgebra.memory
        rw [mem]
      have pidEq : ProgramModel.pid entry=pid := by simp [entry,enter,ProgramModel.pid]
      have memoryEq : KernelAlgebra.memory entry=KernelAlgebra.memory initial := rfl
      rw [nextEq,model,pidEq,memoryEq]
    obtain ⟨final,executedRest,memoryRest,frameRest⟩ :=
      ih (fun q hq => inside q (by simp [hq])) next nextMD
    refine ⟨final,?_,?_,?_⟩
    · change (FP.Structural.exec (KernelAlgebra.backend p)
        (TypedExecution.kernel c.shape cumulative64 offsets64) entry).bind
        (execute p c.shape cumulative64 offsets64 rest)=some final
      rw [executed,Option.bind_some]
      exact executedRest
    · simpa only [sourcePrograms,nextMemory] using memoryRest
    · intro r o outside
      rw [frameRest r o outside,mem]
      exact program_frame c valid p entry inputMD boundB boundH r o outside

theorem complete (c : Config) (valid : c.Valid) (p : Ops α) (cumulative64 offsets64 : Bool)
    (initial : State α) (md : ProgramModel.MetadataBinding c initial) :
    ∃ final,run p c cumulative64 offsets64 initial=some final ∧
      KernelAlgebra.memory final=source c p (KernelAlgebra.memory initial) ∧
      (∀ w,c.Active w → ∀ second,
        (final.mem (KernelAlgebra.regionName c.shape.output) (c.cell w second)).read .real=
          component (pairValues c p w (KernelAlgebra.memory initial)) second) ∧
      (∀ r o,Outside c r o → final.mem r o=initial.mem r o) := by
  by_cases empty : c.shape.total=0 ∨ c.shape.half=0
  · have inactive : ∀ w,¬c.Active w := by
      intro w hw
      rcases empty with total|half
      · have bounded := row_bound c valid w.batch w.row hw.1 hw.2.1
        omega
      · have pair := hw.2.2.2
        omega
    have scheduleNil : schedule c=[] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro w hw
      exact inactive w (schedule_active c w hw)
    refine ⟨initial,by simp [run,empty],?_,?_,fun _ _ _ => rfl⟩
    · rw [source_eq_run c valid]
      simp [RotaryTransform.run,scheduleNil,RotaryTransform.execute]
    · intro w hw
      exact False.elim (inactive w hw)
  obtain ⟨final,success,memory,frame⟩ :=
    complete_list c valid p cumulative64 offsets64 (programs c) (fun _ h => h) initial md
  refine ⟨final,by simpa only [run,if_neg empty] using success,memory,?_,frame⟩
  intro w hw second
  change KernelAlgebra.memory final c.shape.output (c.cell w second)=_
  rw [memory,show sourcePrograms c p (programs c) (KernelAlgebra.memory initial)=
    source c p (KernelAlgebra.memory initial) from rfl,source_eq_run c valid]
  exact run_readback c valid p (KernelAlgebra.memory initial) w hw second

#axiomsClean complete_list
#axiomsClean complete

/-- Every permutation of the complete launch grid has the same initial-input
rotation values and complete tagged-memory frame. Empty launches stay empty. -/
theorem complete_order (c : Config) (valid : c.Valid) (p : Ops α)
    (cumulative64 offsets64 : Bool) (order : List Program)
    (permutation : (programs c).Perm order) (initial : State α)
    (md : ProgramModel.MetadataBinding c initial) :
    ∃ final,runOrder p c cumulative64 offsets64 order initial=some final ∧
      KernelAlgebra.memory final=source c p (KernelAlgebra.memory initial) ∧
      (∀ w,c.Active w → ∀ second,
        (final.mem (KernelAlgebra.regionName c.shape.output) (c.cell w second)).read .real=
          component (pairValues c p w (KernelAlgebra.memory initial)) second) ∧
      (∀ r o,Outside c r o → final.mem r o=initial.mem r o) := by
  by_cases empty : c.shape.total=0 ∨ c.shape.half=0
  · simpa only [runOrder,run,if_pos empty] using
      complete c valid p cumulative64 offsets64 initial md
  obtain ⟨final,success,memory,frame⟩ := complete_list c valid p cumulative64 offsets64
    order (fun _ h => permutation.mem_iff.mpr h) initial md
  rw [sourcePrograms_order c valid p order permutation] at memory
  refine ⟨final,by simpa only [runOrder,if_neg empty] using success,memory,?_,frame⟩
  intro w hw second
  change KernelAlgebra.memory final c.shape.output (c.cell w second)=_
  rw [memory,source_eq_run c valid]
  exact run_readback c valid p (KernelAlgebra.memory initial) w hw second

#axiomsClean complete_order
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedGrid
