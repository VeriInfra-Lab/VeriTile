import bench.tritonbench_g.layernorm_fwd_triton.IO
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.PairExecution
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.PublicIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairIO
open VeriTile.Triton FP.Structural Address KernelPublicIO
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Transport a complete row execution through equal input memory and undef
values. Invocation parameters are rebound before each row; private registers
from earlier rows do not enter the source reference. -/
theorem invocation_transport (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (row : Row shape) (left right final : State α)
    (memory : left.mem=right.mem) (undef : left.undef=right.undef)
    (ran : (invocation M shape st eps row).run left=some final) :
    ∃ other, (candidateInvocation M shape st eps row).run right=some other ∧
      other.mem=final.mem ∧ other.undef=final.undef := by
  let p := Source.parameters st shape.columns
  let ls := (invocation M shape st eps row).initial left
  let rs := (candidateInvocation M shape st eps row).initial right
  have leps : ls.regs .real [] "eps"=some (fun _ => eps) := by
    simp [ls,invocation,Invocation.initial,bindRegisters]
  have reps : rs.regs .real [] "eps"=some (fun _ => eps) := by
    simp [rs,candidateInvocation,invocation,Invocation.initial,bindRegisters]
  obtain ⟨old,oldRun,oldMem,oldUndef⟩ := Source.complete_run_undef M p ls eps leps
    (by simp [p,Source.parameters]) (by simp [p,Source.parameters])
  have oldEq : old=final := Option.some.inj (oldRun.symm.trans ran)
  subst old
  obtain ⟨other,otherRun,otherMem,otherUndef⟩ := PairExecution.candidate_complete M p rs eps reps
    (by simp [p,Source.parameters]) (by simp [p,Source.parameters])
  have reference : Source.referenceMem M p ls=Source.referenceMem M p rs := by
    apply Source.referenceMem_congr
    all_goals simp [ls,rs,candidateInvocation,invocation,Invocation.initial,bindRegisters,memory,undef]
  refine ⟨other,otherRun,otherMem.trans (reference.symm.trans oldMem.symm),?_⟩
  calc
    other.undef=right.undef := by simpa [rs,candidateInvocation,invocation,Invocation.initial,bindRegisters] using otherUndef
    _=left.undef := undef.symm
    _=final.undef := by simpa [ls,invocation,Invocation.initial,bindRegisters] using oldUndef.symm

theorem schedule_transport (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (rows : List (Row shape)) (left right final : State α)
    (memory : left.mem=right.mem) (undef : left.undef=right.undef)
    (ran : runSchedule (rows.map (invocation M shape st eps)) left=some final) :
    ∃ other, runSchedule (rows.map (candidateInvocation M shape st eps)) right=some other ∧
      other.mem=final.mem ∧ other.undef=final.undef := by
  induction rows generalizing left right final with
  | nil =>
    have eq : left=final := by simpa [runSchedule] using ran
    subst final
    exact ⟨right,rfl,memory.symm,undef.symm⟩
  | cons row rest ih =>
    simp only [List.map_cons,runSchedule] at ran
    cases headRun : (invocation M shape st eps row).run left with
    | none => simp [headRun] at ran
    | some middle =>
      simp only [headRun,Option.bind_some] at ran
      obtain ⟨next,nextRun,nextMem,nextUndef⟩ := invocation_transport M shape st eps row
        left right middle memory undef headRun
      obtain ⟨other,tailRun,tailMem,tailUndef⟩ := ih middle next final nextMem.symm nextUndef.symm ran
      exact ⟨other,by simp [runSchedule,nextRun,tailRun],tailMem,tailUndef⟩

theorem candidate_safe {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (rows : List (Row shape)) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ row : Row shape,∀ j : Fin shape.columns,xOffset st row j.val<bounds "X")
    (weight : ∀ row : Row shape,∀ j : Fin shape.columns,wOffset st row j.val<bounds "W")
    (output : ∀ row : Row shape,∀ j : Fin shape.columns,yOffset st row j.val<bounds "Y") :
    ScheduleSafe bounds (rows.map (candidateInvocation M shape st eps)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rest ih =>
    constructor
    · apply PairExecution.candidate_safe M (Source.parameters st shape.columns) bounds
        ((candidateInvocation M shape st eps row).initial s) eps
      · simp [candidateInvocation,invocation,Invocation.initial,bindRegisters]
      · exact (by decide : ("X" : RegionName)≠"Y")
      · exact (by decide : ("W" : RegionName)≠"Y")
      · exact fun j hj => input row ⟨j,hj⟩
      · simpa [Source.parameters,Source.wBase,invocation,Invocation.initial,bindRegisters,
          candidateInvocation,wOffset,offset] using fun (j : Nat) (hj : j<shape.columns) => weight row ⟨j,hj⟩
      · exact fun j hj => output row ⟨j,hj⟩
    · intro t _
      exact ih t

/-- Complete sequence/head grid, all admitted strides and all values of the
selected scalar model. Both actual source schedules are safe and successful;
every output and the complete exterior typed-cell frame agree. -/
theorem equivalent (M : Algebra α) (shape : Shape) (st : Strides) (eps : α)
    (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) :
    (io M shape st eps).Equivalent (variant M shape st eps) := by
  intro values _ bounds hb initial binding
  obtain ⟨safe, final, ran, output, frame⟩ :=
    KernelPublicIO.implements M shape st eps injective
      (fun x w j => Graph.output M shape.columns x w eps j) (fun _ _ _ _ => rfl)
      values trivial bounds hb initial binding
  obtain ⟨other,candidateRun,sameMem,_⟩ := schedule_transport M shape st eps
    (Source.programs shape) initial initial final rfl rfl ran
  refine ⟨safe, ?_, final, other, ran, candidateRun, ?_, frame, ?_⟩
  · exact candidate_safe M shape st eps _ initial bounds
      (fun row j => hb.2.1 0 (row.1,row.2,j,()) trivial)
      (fun row j => hb.2.1 1 (row.1,row.2,j,()) trivial)
      (fun row j => hb.2.2.1 0 (row.1,row.2,j,()) trivial)
  · intro o j active
    change mapCell id (final.mem _ _) = mapCell id (other.mem _ _)
    rw [sameMem]
  · intro r offset outside
    rw [sameMem]
    exact frame r offset outside

#axiomsClean invocation_transport
#axiomsClean schedule_transport
#axiomsClean candidate_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairIO
