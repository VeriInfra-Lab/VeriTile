import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.FirstInputs
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.FirstSafety

namespace VeriTile.Bench.Repairs.MatmulDequantize.FirstIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.MatmulDequantize
open FPTypedFirst (Parameters steps Dot)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

@[simp] theorem initial_eq (M : Algebra A) (D : Dot A) (p : Parameters) (pid : Nat) (s : State A) :
    (invocation M D p pid).initial s=GridExecution.atProgram s pid := rfl

theorem schedule_run (M : Algebra A) (D : Dot A) (p : Parameters) (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map (invocation M D p)) s=Grid.run (FirstGrid.program M D p valid ha hb hs hz) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,initial_eq,
      Grid.run,GridExecution.sequence,FirstGrid.program]
    cases h : exec (FPTypedFirst.backend M D) (FPTypedFirst.source p) (GridExecution.atProgram s pid) <;>
      simp [invocation,h,ih,Grid.run,FirstGrid.program]

theorem source_safe (M : Algebra A) (D : Dot A) (p : Parameters) (vals : Inputs A p)
    (bounds : RegionName → Nat) (backed : (program M D p []).InBounds bounds vals)
    (pid : Nat) (legal : pid<Geometry.count (FirstGrid.geometry p)) (s : State A) :
    (invocation M D p pid).Safe bounds s := by
  apply FirstAccess.source_safe
  constructor
  · intro k hk i hm
    let j : TileIndex (AShape p) := (⟨pid,legal⟩,⟨k,hk⟩,i)
    have h := backed.2.1 ⟨0,by simp [program,memory,inputChannels]⟩ j
      (by simpa [program,memory,inputActive,inputChannels,Fin.cases,aMask,j] using hm)
    exact h
  · intro k hk i
    exact backed.2.1 ⟨1,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk⟩,i)
      (by change True; trivial)
  · intro k hk i
    have hk' : k<max (steps p) 1 := by rcases hk with hk|⟨_,rfl⟩ <;> omega
    exact backed.2.1 ⟨2,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk'⟩,i)
      (by simpa [program,memory,inputActive,inputChannels,Fin.cases,sMask] using hk)
  · intro k hk i
    have hk' : k<max (steps p) 1 := by rcases hk with hk|⟨_,rfl⟩ <;> omega
    exact backed.2.1 ⟨3,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk'⟩,i)
      (by simpa [program,memory,inputActive,inputChannels,Fin.cases,sMask] using hk)
  · intro i hi
    rw [initial_eq,FirstGrid.offset_at]
    exact backed.2.2.1 ⟨0,by simp [program,memory,outputChannels]⟩ (⟨pid,legal⟩,i)
      (by simpa [program,memory,outputChannels,Fin.cases,initial_eq] using hi)

theorem schedule_safe (M : Algebra A) (D : Dot A) (p : Parameters) (vals : Inputs A p)
    (bounds : RegionName → Nat) (backed : (program M D p []).InBounds bounds vals)
    (jobs : List Nat) (legal : ∀ pid∈jobs,pid<Geometry.count (FirstGrid.geometry p))
    (s : State A) : ScheduleSafe bounds (jobs.map (invocation M D p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    exact ⟨source_safe M D p vals bounds backed pid (legal pid (by simp)) s,
      fun t _ => ih (fun q member => legal q (by simp [member])) t⟩

/-- Complete original source-grid execution, source safety, typed outputs,
independent recurrence, and exact exterior frame, with no new native restriction. -/
theorem expression_implements (M : Algebra A) (D : Dot A) (p : Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p)))) :
    (program M D p jobs).Implements (expression M D p) := by
  intro vals _ bounds backed s bound
  obtain ⟨t,ran,values,frame⟩ := FirstGrid.whole M D p valid ha hb hs hz s jobs schedule
  refine ⟨schedule_safe M D p vals bounds backed jobs
    (fun pid member => List.mem_range.mp (schedule.mem_iff.mp member)) s,t,?_,?_,?_⟩
  · exact (schedule_run M D p valid ha hb hs hz jobs s).trans ran
  · intro o j hj
    fin_cases o
    change mapCell id (t.mem p.output (Geometry.offset (FirstGrid.geometry p) j.1.val j.2))=
      Cell.mk .fp16 (expression M D p vals ⟨0,by simp [outputChannels]⟩ j)
    rw [values j.1.val j.1.isLt j.2 hj,expression_binding M D p jobs vals s bound j]
    rfl
  · intro r off outside
    apply frame r off
    by_cases hr : r=p.output
    · right
      intro pid hp i hi
      exact (outside.1 ⟨0,by simp [program,memory,outputChannels]⟩ (⟨pid,hp⟩,i)
        (by simpa [program,memory,outputChannels,Fin.cases] using hi)).resolve_left (fun ne => ne hr)
    · exact Or.inl hr

theorem native_observes (p : Parameters) (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (nativeIO p jobs).Observes (observe p plan isa) (nativeTarget p plan isa) := by
  have h := ProgramIO.Observes.of_implements
    (expression_implements Native.scalar Native.firstDot p valid ha hb hs hz jobs schedule) (observe p plan isa)
  convert h using 1
  funext vals o j
  fin_cases o
  rfl

/-- Every original arbitrary memory, including arbitrary native expressions,
supplies the finite logical inputs at all and only the declared source windows. -/
def inputsOfState (p : Parameters) (s : State A) : Inputs A p :=
  fun i j => (s.mem (inputChannels p)[i].buf (inputAddress p i j)).read (inputChannels p)[i].dtype

theorem inputsOfState_binding (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat) (s : State A) :
    (program M D p jobs).InputBinding (inputsOfState p s) s := by
  intro i j _
  fin_cases i <;> rfl

#axiomsClean expression_implements
#axiomsClean native_observes
#axiomsClean inputsOfState_binding
end
end VeriTile.Bench.Repairs.MatmulDequantize.FirstIO
