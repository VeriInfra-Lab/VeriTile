import bench.repairs.tritonbench_g.swiglu_backward.IO
import bench.repairs.tritonbench_g.swiglu_backward.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.SwigluBackward.IOProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

theorem run_schedule {β : Type} [Inhabited β] (M : Algebra β) (c : Layout)
    (B : Nat) (R : Bool) (tiles : List (Nat × Nat)) (s : State β) :
    runSchedule (tiles.map (IO.invocation M c B R)) s = runTiles M c B R tiles s := by
  induction tiles generalizing s with
  | nil => rfl
  | cons t ts ih =>
    change ((IO.invocation M c B R t).run s).bind _ = (tileExec M c B R t s).bind _
    have head : (IO.invocation M c B R t).run s = tileExec M c B R t s := rfl
    rw [head]
    cases tileExec M c B R t s <;> simp only [Option.bind_none,Option.bind_some,ih]

/-- Every actual grid successor satisfies the declared windows; the proof
follows the same launch sequence and intermediate states as execution. -/
theorem schedule_safe {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α) (c : Layout) (B : Nat) (R : Bool)
    (tiles : List (Nat × Nat)) (valid : ∀ t ∈ tiles, t.1 < c.rows)
    (vals : ChannelValues α (IO.inputs c)) (bounds : RegionBounds)
    (bounded : (IO.swigluIO M encode observe c B R tiles).InBounds bounds vals)
    (s : State β) : ScheduleSafe bounds (tiles.map (IO.invocation M c B R)) s := by
  induction tiles generalizing s with
  | nil => trivial
  | cons t ts ih =>
    refine ⟨?_,fun next _ => ih (fun u hu => valid u (List.mem_cons_of_mem t hu)) bounded next⟩
    change TraceSafe M bounds (c.source B R) (tileState M c B t.1 t.2 s)
    have row := valid t List.mem_cons_self
    apply SourceSafety.source_safe
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

/-- Reuse the established complete-grid result, preserving its arbitrary
permutation, all enabled output channels, and every exterior cell. -/
theorem implements {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α) (f : Channel → α → α → α → α)
    (lower : ∀ ch x y d, observe (scalar M ch (encode x) (encode y) (encode d)) = f ch x y d)
    (c : Layout) (B : Nat) (R : Bool) (positive : 0 < B) (own : Ownership c R)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B)) :
    (IO.swigluIO M encode observe c B R tiles).Implements (IO.reference c f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded s _
    exact schedule_safe M encode observe c B R tiles
      (fun t ht => grid_valid c B t (schedule.mem_iff.mp ht)) vals bounds bounded s
  · intro vals s binding
    obtain ⟨final,ran,values,frame⟩ := grid_result M c B R positive own tiles schedule s
    refine ⟨final,?_,?_,?_⟩
    · exact (run_schedule M c B R tiles s).trans ran
    · intro o j active
      fin_cases o
      all_goals
        have hx := binding 0 j trivial
        have hy := binding 1 j trivial
        have hd := binding 2 j trivial
        change (s.mem c.X (j.1.val*c.sx+j.2.1.val)).read .real = encode (vals 0 j) at hx
        change (s.mem c.Y (j.1.val*c.sy+j.2.1.val)).read .real = encode (vals 1 j) at hy
        change (s.mem c.DOUT (j.1.val*c.sd+j.2.1.val)).read .real = encode (vals 2 j) at hd
      · change mapCell observe (final.mem (c.output .dx) (j.1.val*c.stride .dx+j.2.1.val)) =
          Cell.mk .real (f .dx (vals 0 j) (vals 1 j) (vals 2 j))
        rw [values .dx (by simp [enabled]) _ _ j.1.isLt j.2.1.isLt]
        exact congrArg (Cell.mk .real) (by simpa only [hx,hy,hd] using lower .dx (vals 0 j) (vals 1 j) (vals 2 j))
      · change mapCell observe (final.mem (c.output .dy) (j.1.val*c.stride .dy+j.2.1.val)) =
          Cell.mk .real (f .dy (vals 0 j) (vals 1 j) (vals 2 j))
        rw [values .dy (by simp [enabled]) _ _ j.1.isLt j.2.1.isLt]
        exact congrArg (Cell.mk .real) (by simpa only [hx,hy,hd] using lower .dy (vals 0 j) (vals 1 j) (vals 2 j))
      · change mapCell observe (final.mem (c.output .out) (j.1.val*c.stride .out+j.2.1.val)) =
          Cell.mk .real (f .out (vals 0 j) (vals 1 j) (vals 2 j))
        rw [values .out (Or.inr active) _ _ j.1.isLt j.2.1.isLt]
        exact congrArg (Cell.mk .real) (by simpa only [hx,hy,hd] using lower .out (vals 0 j) (vals 1 j) (vals 2 j))
    · intro r offset outside
      apply frame
      intro ch on
      by_cases same : r = c.output ch
      · right
        intro row col hr hc equal
        cases ch with
        | dx =>
          rcases outside.1 0 (⟨row,hr⟩,⟨col,hc⟩,()) trivial with h | h
          · exact h same
          · exact h equal
        | dy =>
          rcases outside.1 1 (⟨row,hr⟩,⟨col,hc⟩,()) trivial with h | h
          · exact h same
          · exact h equal
        | out =>
          rcases outside.1 2 (⟨row,hr⟩,⟨col,hc⟩,()) (by simpa [enabled] using on) with h | h
          · exact h same
          · exact h equal
      · exact Or.inl same

#axiomsClean run_schedule
#axiomsClean schedule_safe
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.IOProof
