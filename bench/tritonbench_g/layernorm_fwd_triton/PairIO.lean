import bench.tritonbench_g.layernorm_fwd_triton.IO
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.PairExecution
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.PublicIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairIO
open VeriTile.Triton FP.Structural Address KernelPublicIO
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem invocation_eq (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (row : Row shape) (s : State α) :
    (candidateInvocation M shape st eps row).run s = (invocation M shape st eps row).run s := by
  apply (PairExecution.exact M (Source.parameters st shape.columns)
    ((invocation M shape st eps row).initial s) eps ?_ (by simp [Source.parameters]) (by simp [Source.parameters])).symm
  simp [invocation, Invocation.initial, bindRegisters]

theorem schedule_eq (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (rows : List (Row shape)) (s : State α) :
    runSchedule (rows.map (candidateInvocation M shape st eps)) s =
      runSchedule (rows.map (invocation M shape st eps)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    simp only [List.map_cons, runSchedule, invocation_eq]
    cases h : (invocation M shape st eps row).run s <;> simp [h, ih]
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
  refine ⟨safe, ?_, final, final, ran, ?_, ?_, frame, frame⟩
  · exact candidate_safe M shape st eps _ initial bounds
      (fun row j => hb.2.1 0 (row.1,row.2,j,()) trivial)
      (fun row j => hb.2.1 1 (row.1,row.2,j,()) trivial)
      (fun row j => hb.2.2.1 0 (row.1,row.2,j,()) trivial)
  · exact (schedule_eq M shape st eps _ initial).trans ran
  · intro o j active
    rfl

#axiomsClean invocation_eq
#axiomsClean schedule_eq
#axiomsClean candidate_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairIO
