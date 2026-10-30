import bench.tritonbench_g.matmul_dequantize.IO
import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant32Safety

namespace VeriTile.Bench.Repairs.MatmulDequantize.D32IO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open Dequant32 (Parameters)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

@[simp] theorem initial_eq (M : Algebra A) (p : Parameters) (pid : Nat) (s : State A) :
    (invocation M p pid).initial s = DequantGrid32.atProgram p s pid := rfl

theorem schedule_run (M : Algebra A) (p : Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map (invocation M p)) s=Grid.run (DequantGrid32.program M p valid hb hs hz) jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,initial_eq,
      Grid.run,GridExecution.sequence,DequantGrid32.program]
    cases h : exec M (Dequant32.source p) (DequantGrid32.atProgram p s pid) <;> simp [invocation,h,ih,Grid.run,DequantGrid32.program]

theorem value_binding (M : Algebra A) (p : Parameters) (jobs : List Nat)
    (vals : Inputs A p) (s : State A) (bound : (program M p jobs).InputBinding vals s)
    (j : Lane p) (hj : active p j) :
    Dequant32.value M p (DequantGrid32.atProgram p s j.1.val) j.2=
      expression M p vals ⟨0,by simp [outputChannels]⟩ j := by
  have w := bound ⟨0,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  have z := bound ⟨1,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  have c := bound ⟨2,by simp [program,memory,inputChannels]⟩ j (by simpa [program,memory,inputActive,inputChannels,Fin.cases] using hj)
  change (s.mem p.weight (weightAddress p j)).read .int=vals ⟨0,by simp [inputChannels]⟩ j at w
  change (s.mem p.zero (zeroAddress p j)).read .int=vals ⟨1,by simp [inputChannels]⟩ j at z
  change (s.mem p.scale (scaleAddress p j)).read .fp32=vals ⟨2,by simp [inputChannels]⟩ j at c
  change M.binary none .fp32 .mul
    (M.cast (some .fp32) .real .fp32 (M.fromInt (some .fp32)
      (Dequant32.nibble ((s.mem p.weight (weightAddress p j)).read .int) (row p j)-
       Dequant32.nibble ((s.mem p.zero (zeroAddress p j)).read .int) (column p j))))
    ((s.mem p.scale (scaleAddress p j)).read .fp32)=_
  rw [w,z,c]
  rfl

theorem source_safe (M : Algebra A) (p : Parameters) (vals : Inputs A p)
    (bounds : RegionName → Nat) (backed : (program M p []).InBounds bounds vals)
    (pid : Nat) (legal : pid<Geometry.count (DequantGrid32.geometry p)) (s : State A) :
    (invocation M p pid).Safe bounds s := by
  apply Dequant32Access.source_safe
  constructor
  · intro i hi
    exact backed.2.1 ⟨0,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,i) (by simpa [program,memory,inputActive,inputChannels,outputChannels,Fin.cases,active,Invocation.initial,invocation,bindRegisters] using hi)
  · intro i hi
    exact backed.2.1 ⟨1,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,i) (by simpa [program,memory,inputActive,inputChannels,outputChannels,Fin.cases,active,Invocation.initial,invocation,bindRegisters] using hi)
  · intro i hi
    exact backed.2.1 ⟨2,by simp [program,memory,inputChannels]⟩ (⟨pid,legal⟩,i) (by simpa [program,memory,inputActive,inputChannels,outputChannels,Fin.cases,active,Invocation.initial,invocation,bindRegisters] using hi)
  · intro i hi
    exact backed.2.2.1 ⟨0,by simp [program,memory,outputChannels]⟩ (⟨pid,legal⟩,i) (by simpa [program,memory,inputActive,inputChannels,outputChannels,Fin.cases,active,Invocation.initial,invocation,bindRegisters] using hi)

theorem schedule_safe (M : Algebra A) (p : Parameters) (vals : Inputs A p)
    (bounds : RegionName → Nat) (backed : (program M p []).InBounds bounds vals)
    (jobs : List Nat) (legal : ∀ pid∈jobs,pid<Geometry.count (DequantGrid32.geometry p))
    (s : State A) : ScheduleSafe bounds (jobs.map (invocation M p)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    exact ⟨source_safe M p vals bounds backed pid (legal pid (by simp)) s,
      fun t _ => ih (fun q member => legal q (by simp [member])) t⟩

/-- The logical relation uses precisely the original complete arbitrary-order
source grid, all active cells and its exact exterior frame. -/
theorem expression_implements (M : Algebra A) (p : Parameters)
    (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p)))) :
    (program M p jobs).Implements (expression M p) := by
  intro vals _ bounds backed s bound
  obtain ⟨t,ran,values,frame⟩ := DequantGrid32.whole M p valid hb hs hz s jobs schedule
  refine ⟨schedule_safe M p vals bounds backed jobs
    (fun pid member => List.mem_range.mp (schedule.mem_iff.mp member)) s,t,?_,?_,?_⟩
  · exact (schedule_run M p valid hb hs hz jobs s).trans ran
  · intro o j hj
    fin_cases o
    change mapCell id (t.mem p.output (Geometry.offset (DequantGrid32.geometry p) j.1.val j.2))=
      Cell.mk .fp32 (expression M p vals ⟨0,by simp [outputChannels]⟩ j)
    rw [values j.1.val j.1.isLt j.2 hj,value_binding M p jobs vals s bound j hj]
    rfl
  · intro r off outside
    apply frame r off
    by_cases hr : r=p.output
    · right
      intro pid hp i hi
      exact (outside.1 ⟨0,by simp [program,memory,outputChannels]⟩ (⟨pid,hp⟩,i) (by simpa [program,memory,outputChannels,Fin.cases,active] using hi)).resolve_left
        (fun ne => ne hr)
    · exact Or.inl hr

theorem real_implements (p : Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p)))) :
    (realIO p jobs).Implements (mathematical p) := by
  have formula : expression Real.scalar p=mathematical p := by
    funext vals o j
    fin_cases o
    rfl
  change (program Real.scalar p jobs).Implements (mathematical p)
  rw [←formula]
  exact expression_implements Real.scalar p valid hb hs hz jobs schedule

theorem native_observes (p : Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (nativeIO p jobs).Observes (observe p plan isa) (nativeTarget p plan isa) := by
  have h := ProgramIO.Observes.of_implements
    (expression_implements Native.scalar p valid hb hs hz jobs schedule) (observe p plan isa)
  convert h using 1
  funext vals o j
  fin_cases o
  rfl

/-- Every original arbitrary initial state induces finite logical inputs;
this adapter does not narrow symbolic payloads or admissible input aliasing. -/
def inputsOfState (p : Parameters) (s : State A) : Inputs A p :=
  fun i j => (s.mem (inputChannels p)[i].buf (inputAddress p i j)).read (inputChannels p)[i].dtype

theorem inputsOfState_binding (M : Algebra A) (p : Parameters) (jobs : List Nat) (s : State A) :
    (program M p jobs).InputBinding (inputsOfState p s) s := by
  intro i j _
  fin_cases i <;> rfl

#axiomsClean expression_implements
#axiomsClean real_implements
#axiomsClean native_observes
#axiomsClean inputsOfState_binding
end
end VeriTile.Bench.Repairs.MatmulDequantize.D32IO
