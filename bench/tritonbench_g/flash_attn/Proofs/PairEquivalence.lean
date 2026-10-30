import bench.tritonbench_g.flash_attn.PairIO
import bench.tritonbench_g.flash_attn.Proofs.IndexPair
import bench.tritonbench_g.flash_attn.Proofs.IONative

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PairIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
open scoped VeriTile.Spec
noncomputable section
variable {A : Type} [Inhabited A]

theorem invocation_runs (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A)
    (job : ActualSource.Program) (s : State A) :
    (commutedInvocation M dot x scale job).run s = (PublicIO.invocation M dot x scale job).run s :=
  IndexPair.runs _ _ _

theorem invocation_safe (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A)
    (job : ActualSource.Program) (bounds : RegionName → Nat) (s : State A) :
    (commutedInvocation M dot x scale job).Safe bounds s ↔
      (PublicIO.invocation M dot x scale job).Safe bounds s :=
  IndexPair.safe _ _ _ _

theorem schedule_runs (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A)
    (jobs : List ActualSource.Program) (s : State A) :
    runSchedule (jobs.map (commutedInvocation M dot x scale)) s =
      runSchedule (jobs.map (PublicIO.invocation M dot x scale)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih => simp only [List.map_cons,runSchedule,invocation_runs,ih]

theorem schedule_safe (M : Algebra A) (dot : Dot A) (x : Layout) (scale : A)
    (jobs : List ActualSource.Program) (bounds : RegionName → Nat) (s : State A) :
    ScheduleSafe bounds (jobs.map (commutedInvocation M dot x scale)) s ↔
      ScheduleSafe bounds (jobs.map (PublicIO.invocation M dot x scale)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih => simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_runs,ih]

theorem implements (p : Program x) (I : KernelNative.Instructions) (scale : Checked.ScalarABI) :
    (p.io I scale).Implements (Logical.nativeFormula I x scale) := by
  rcases p with ⟨swapped⟩
  cases swapped
  · exact PublicIO.native_implements I x scale
  · intro xs hdomain bounds hb initial hin
    obtain ⟨hs,final,he,ho,hf⟩ := PublicIO.native_implements I x scale xs hdomain bounds hb initial hin
    refine ⟨?_,final,?_,ho,hf⟩
    · exact (schedule_safe _ _ x _ _ bounds initial).mpr hs
    · change runSchedule ((programs (parameters x)).map (commutedInvocation _ _ x _)) initial = some final
      rw [schedule_runs]
      exact he

/-- Reuses the full native IO result only as a proved execution/output/frame
lemma. The primary conclusion compares the two actual complete programs. -/
theorem equivalent (x : Layout) : original x ≡[assumptions] candidate x := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := original x) (rhs := candidate x) rfl rfl
  intro I scale
  exact ProgramIO.EquivalentOn.of_implementsOn (implements (original x) I scale)
    (implements (candidate x) I scale)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PairIO
