import bench.tritonbench_g.matmul_dequantize.Proofs.SecondInputs
import bench.tritonbench_g.matmul_dequantize.Proofs.SecondSafety

namespace VeriTile.Bench.Repairs.MatmulDequantize.SecondIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open FPTypedSecond (Parameters Dot)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

@[simp] theorem initial_eq (M : Algebra A) (D : Dot A) (p : Parameters) (pid : Nat) (s : State A) :
    (invocation M D p pid).initial s=GridExecution.atProgram s pid := rfl

theorem schedule_run (M : Algebra A) (D : Dot A) (p : Parameters) (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map (invocation M D p)) s=Grid.run (SecondGrid.program M D p valid ha hb hs hz) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,initial_eq,
      Grid.run,GridExecution.sequence,SecondGrid.program]
    cases h : exec (FPTypedSecond.backend M D) (FPTypedSecond.source p) (GridExecution.atProgram s pid) <;>
      simp [invocation,h,ih,Grid.run,SecondGrid.program]

theorem source_safe (M : Algebra A) (D : Dot A) (p : Parameters) (vals : Inputs A p)
    (bounds : RegionName → Nat) (backed : (program M D p []).InBounds bounds vals)
    (pid : Nat) (legal : pid<Geometry.count (SecondGrid.geometry p)) (s : State A) :
    (invocation M D p pid).Safe bounds s := by
  apply SecondAccess.source_safe
  constructor
  · intro k hk i hm
    have hm' : i.2.1.val+k*p.blockInner<p.inner := by
      simpa only [initial_eq,FPTypedSecond.innerActive,inner_at] using hm
    have h := backed.2.1 ⟨0,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk⟩,i)
      (by change i.2.1.val+k*p.blockInner<p.inner; exact hm')
    change aAddress p pid k i.1.val i.2.1.val<bounds p.input at h
    simpa only [initial_eq,FPTypedSecond.aOffset,inner_at] using h
  · intro k hk i hm
    have hm' : i.1.val+k*p.blockInner<p.inner := by
      simpa only [initial_eq,FPTypedSecond.innerActive,inner_at] using hm
    have h := backed.2.1 ⟨1,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk⟩,i)
      (by change i.1.val+k*p.blockInner<p.inner; exact hm')
    change bAddress p pid k i.1.val i.2.1.val<bounds p.weight at h
    simpa only [initial_eq,FPTypedSecond.bOffset,inner_at] using h
  · intro k hk i hm
    have hm' : i.1.val+k*p.blockInner<p.inner := by
      simpa only [initial_eq,FPTypedSecond.innerActive,inner_at] using hm
    have h := backed.2.1 ⟨2,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk⟩,i)
      (by change i.1.val+k*p.blockInner<p.inner; exact hm')
    change sAddress p pid k i.1.val i.2.1.val<bounds p.scale at h
    simpa only [initial_eq,FPTypedSecond.scaleOffset,inner_at] using h
  · intro k hk i hm
    have hm' : i.1.val+k*p.blockInner<p.inner := by
      simpa only [initial_eq,FPTypedSecond.innerActive,inner_at] using hm
    have h := backed.2.1 ⟨3,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,⟨k,hk⟩,i)
      (by change i.1.val+k*p.blockInner<p.inner; exact hm')
    change zAddress p pid k i.1.val i.2.1.val<bounds p.zero at h
    simpa only [initial_eq,FPTypedSecond.zeroOffset,inner_at] using h
  · intro i hi
    rw [initial_eq,SecondGrid.offset_at]
    exact backed.2.2.1 ⟨0,by simp [program,memory,outputChannels]⟩ (⟨pid,legal⟩,i)
      (by simpa [program,memory,outputChannels,Fin.cases,initial_eq] using hi)

theorem schedule_safe (M : Algebra A) (D : Dot A) (p : Parameters) (vals : Inputs A p)
    (bounds : RegionName → Nat) (backed : (program M D p []).InBounds bounds vals)
    (jobs : List Nat) (legal : ∀ pid∈jobs,pid<Geometry.count (SecondGrid.geometry p))
    (s : State A) : ScheduleSafe bounds (jobs.map (invocation M D p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    exact ⟨source_safe M D p vals bounds backed pid (legal pid (by simp)) s,
      fun t _ => ih (fun q member => legal q (by simp [member])) t⟩

/-- Complete original source-grid execution, source safety, typed outputs,
independent recurrence, and exact exterior frame, with no new native restriction. -/
theorem expression_implements (M : Algebra A) (D : Dot A) (p : Parameters)
    (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p)))) :
    (program M D p jobs).Implements (expression M D p) := by
  intro vals _ bounds backed s bound
  obtain ⟨t,ran,values,frame⟩ := SecondGrid.whole M D p valid ha hb hs hz s jobs schedule
  refine ⟨schedule_safe M D p vals bounds backed jobs
    (fun pid member => List.mem_range.mp (schedule.mem_iff.mp member)) s,t,?_,?_,?_⟩
  · exact (schedule_run M D p valid ha hb hs hz jobs s).trans ran
  · intro o j hj
    fin_cases o
    change mapCell id (t.mem p.output (Geometry.offset (SecondGrid.geometry p) j.1.val j.2))=
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

theorem native_observes (p : Parameters) (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (nativeIO p jobs).Observes (observe p plan isa) (nativeTarget p plan isa) := by
  have h := ProgramIO.Observes.of_implements
    (expression_implements Native.scalar Native.secondDot p valid ha hb hs hz jobs schedule) (observe p plan isa)
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
end VeriTile.Bench.Repairs.MatmulDequantize.SecondIO
