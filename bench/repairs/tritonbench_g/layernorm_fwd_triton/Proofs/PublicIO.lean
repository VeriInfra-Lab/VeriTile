import bench.repairs.tritonbench_g.layernorm_fwd_triton.IO
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceSafety
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelGrid
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelPublicIO
open VeriTile.Triton FP.Structural Address
noncomputable section

theorem schedule_run {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (rows : List (Row shape)) (s : State α) :
    runSchedule (rows.map (invocation M shape st eps)) s =
      Source.runRows M (Source.parameters st shape.columns) eps rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    simp only [List.map_cons,runSchedule,Source.runRows,invocation,Invocation.run,
      Invocation.initial,bindRegisters,List.foldl_cons,List.foldl_nil,Option.getD_none,
      Source.atRow]
    congr 1
    funext t
    exact ih t

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (rows : List (Row shape)) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ row : Row shape,∀ j : Fin shape.columns,xOffset st row j.val<bounds "X")
    (weight : ∀ row : Row shape,∀ j : Fin shape.columns,wOffset st row j.val<bounds "W")
    (output : ∀ row : Row shape,∀ j : Fin shape.columns,yOffset st row j.val<bounds "Y") :
    ScheduleSafe bounds (rows.map (invocation M shape st eps)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rest ih =>
    constructor
    · apply KernelSafety.trace_safe M (Source.parameters st shape.columns) bounds
        ((invocation M shape st eps row).initial s) eps
      · simp [invocation,Invocation.initial,bindRegisters]
      · exact (by decide : ("X" : RegionName)≠"Y")
      · exact (by decide : ("W" : RegionName)≠"Y")
      · exact fun j hj => input row ⟨j,hj⟩
      · simpa [Source.parameters,Source.wBase,invocation,Invocation.initial,bindRegisters,
          wOffset,offset] using fun (j : Nat) (hj : j<shape.columns) => weight row ⟨j,hj⟩
      · exact fun j hj => output row ⟨j,hj⟩
    · intro t _
      exact ih t

/-- Existing complete-grid execution plus logical input binding and actual
three-loop source safety. The scalar identity is discharged by the existing
Real/native formula theorem in each public entry. -/
theorem implements {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val))
    (f : (Nat → α) → (Nat → α) → Nat → α)
    (scalar : ∀ x w j,j<shape.columns → Graph.output M shape.columns x w eps j=f x w j) :
    (io M shape st eps).Implements (target M shape st eps f) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb s _
    exact schedule_safe M shape st eps _ s bounds
      (fun row j => hb.2.1 0 (row.1,row.2,j,()) trivial)
      (fun row j => hb.2.1 1 (row.1,row.2,j,()) trivial)
      (fun row j => hb.2.2.1 0 (row.1,row.2,j,()) trivial)
  · intro xs s input
    obtain ⟨t,ran,effect⟩ := KernelGrid.complete M injective eps s
    refine ⟨t,?_,?_,?_⟩
    · exact (schedule_run M shape st eps _ s).trans ran
    · intro o
      fin_cases o
      intro j _
      change mapCell id (t.mem "Y" (yOffset st (rowOf shape j) j.2.2.1.val)) =
        Cell.mk .fp32 (f (xRow M shape st eps xs (rowOf shape j))
          (wRow M shape st eps xs (rowOf shape j)) j.2.2.1.val)
      rw [effect.output (rowOf shape j) (KernelGrid.mem_programs _) j.2.2.1]
      change Cell.mk .fp32 (Graph.output M shape.columns _ _ eps j.2.2.1.val) = _
      rw [Graph.output_congr M shape.columns _ (xRow M shape st eps xs (rowOf shape j))
        _ (wRow M shape st eps xs (rowOf shape j)) eps]
      · rw [scalar _ _ _ j.2.2.1.isLt]
      · intro k hk
        have h := input 0 (j.1,j.2.1,⟨k,hk⟩,()) trivial
        change (s.mem "X" (xOffset st (rowOf shape j) k)).read .fp32=xs 0 (j.1,j.2.1,⟨k,hk⟩,()) at h
        simpa only [KernelGrid.xValues,xRow,dif_pos hk,rowOf] using h
      · intro k hk
        have h := input 1 (j.1,j.2.1,⟨k,hk⟩,()) trivial
        change (s.mem "W" (wOffset st (rowOf shape j) k)).read .fp32=xs 1 (j.1,j.2.1,⟨k,hk⟩,()) at h
        simpa only [KernelGrid.wValues,wRow,dif_pos hk,rowOf] using h
      · exact j.2.2.1.isLt
    · intro r a outside
      apply effect.frame r a
      by_cases same : r="Y"
      · right
        intro row _ j equal
        exact (outside.1 0 (row.1,row.2,j,()) trivial).elim (fun hn => hn same) (fun hn => hn equal)
      · exact Or.inl same

#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelPublicIO
