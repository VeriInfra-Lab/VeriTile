import bench.tritonbench_g.swiglu_backward.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.SwigluBackward.GridEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Safety follows the candidate's actual intermediate states on the full grid. -/
theorem schedule_safe {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α) (c : Layout) (B : Nat) (R : Bool)
    (tiles : List (Nat × Nat)) (valid : ∀ t ∈ tiles, t.1 < c.rows)
    (vals : ChannelValues α (IO.inputs c)) (bounds : RegionBounds)
    (bounded : (IO.swigluIO M encode observe c B R tiles).InBounds bounds vals)
    (s : State β) : ScheduleSafe bounds (tiles.map (IO.commutedInvocation M c B R)) s := by
  induction tiles generalizing s with
  | nil => trivial
  | cons t ts ih =>
    refine ⟨?_,fun next _ => ih (fun u hu => valid u (List.mem_cons_of_mem t hu)) bounded next⟩
    change TraceSafe M bounds
      (commutedKernel c.X c.Y c.DOUT c.OUT c.DX c.DY
        c.sx c.sy c.sd c.so c.sdx c.sdy c.cols B R) (tileState M c B t.1 t.2 s)
    have row := valid t List.mem_cons_self
    apply SourceEquivalence.candidate_safe
    · intro j active
      exact bounded.2.1 0 (⟨t.1,row⟩,⟨t.2*B+j.val,active⟩,()) trivial
    · intro j active
      exact bounded.2.1 1 (⟨t.1,row⟩,⟨t.2*B+j.val,active⟩,()) trivial
    · intro j active
      exact bounded.2.1 2 (⟨t.1,row⟩,⟨t.2*B+j.val,active⟩,()) trivial
    · intro ch on j active
      cases ch with
      | dx => exact bounded.2.2.1 0 (⟨t.1,row⟩,⟨t.2*B+j.val,active⟩,()) trivial
      | dy => exact bounded.2.2.1 1 (⟨t.1,row⟩,⟨t.2*B+j.val,active⟩,()) trivial
      | out =>
        exact bounded.2.2.1 2 (⟨t.1,row⟩,⟨t.2*B+j.val,active⟩,())
          (by simpa [enabled] using on)

/-- Reuse the original full source execution and readback, transporting only
exact integer-address run equality and the candidate access-safety proof. -/
theorem candidate_implements {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α)
    (c : Layout) (B : Nat) (R : Bool) (positive : 0<B) (own : Ownership c R)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B)) :
    (IO.commuted M encode observe c B R tiles).toProgramIO.Implements
      (IO.reference c (fun ch x y d => observe (scalar M ch (encode x) (encode y) (encode d)))) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨_,final,ran,outputs,frame⟩ :=
    IOProof.implements M encode observe _ (fun _ _ _ _ => rfl)
      c B R positive own tiles schedule vals domain bounds bounded initial binding
  refine ⟨schedule_safe M encode observe c B R tiles
    (fun t ht => grid_valid c B t (schedule.mem_iff.mp ht)) vals bounds bounded initial,
    final,?_,outputs,frame⟩
  exact (SourceEquivalence.schedule_run M c B R tiles initial).symm.trans ran

theorem equivalent {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α)
    (c : Layout) (B : Nat) (R : Bool) (positive : 0<B) (own : Ownership c R)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B)) :
    (IO.swigluIO M encode observe c B R tiles).Equivalent
      (IO.commuted M encode observe c B R tiles) := by
  apply ProgramIO.EquivalentOn.of_implementsOn
    (IOProof.implements M encode observe _ (fun _ _ _ _ => rfl)
      c B R positive own tiles schedule)
  exact candidate_implements M encode observe c B R positive own tiles schedule

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.SwigluBackward.GridEquivalence
