import bench.repairs.tritonbench_g.swiglu_fwd.IO
import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.SourceSafety
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.SwigluFwd.KernelPublic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- Compose one source execution proof with a scalar interpretation theorem.
The scalar equation is discharged by the chosen Real/native interpretation;
no new numerical law is introduced by this adapter. -/
theorem implements {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α) (f : α → α → α)
    (scalarLaw : ∀ x y, observe (scalar M (encode x) (encode y))=f x y)
    (c : Layout) (B row tile : Nat) :
    (io M encode observe c B row tile).Implements (target M encode observe c B row tile f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb s _
    rw [show (io M encode observe c B row tile).schedule vals = [_] from rfl,
      scheduleSafe_singleton]
    apply KernelProof.trace_safe
    · intro j hj
      exact hb.2.1 0 (j,()) hj
    · intro j hj
      exact hb.2.1 1 (j,()) hj
    · intro j hj
      exact hb.2.2.1 0 (j,()) hj
  · intro vals s input
    obtain ⟨final,run,values,frame⟩ := KernelProof.run M c B row tile s
    refine ⟨final,?_,?_,?_⟩
    · simpa only [io,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,tileState] using run
    · intro o
      fin_cases o
      intro j active
      have x := input 0 j active
      have y := input 1 j active
      change mapCell observe (final.mem c.OUT (row*c.so+(tile*B+j.1.val))) =
        Cell.mk .real (f (vals 0 j) (vals 1 j))
      rw [values j.1 active]
      change Cell.mk .real (observe (scalar M _ _)) = _
      rw [show (s.mem c.X (row*c.sx+(tile*B+j.1.val))).read .real = encode (vals 0 j) from x,
        show (s.mem c.Y (row*c.sy+(tile*B+j.1.val))).read .real = encode (vals 1 j) from y,
        scalarLaw]
    · intro r off outside
      apply frame r off
      by_cases eq : r=c.OUT
      · right
        intro j active he
        have h := outside.1 0 j active
        exact h.elim (fun hn => hn eq) (fun hn => hn he)
      · exact Or.inl eq

#axiomsClean implements
end
end VeriTile.Bench.Repairs.SwigluFwd.KernelPublic
