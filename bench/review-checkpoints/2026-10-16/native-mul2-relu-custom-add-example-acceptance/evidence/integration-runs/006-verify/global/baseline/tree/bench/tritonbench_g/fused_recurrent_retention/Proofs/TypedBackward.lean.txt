import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardFirst
import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardMiddle
import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardSecond
import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedForward

/-! Whole fp32 backward typed success, with both original loops and the full
middle section. No numerical backend laws or successful intermediate states
are assumptions. The explicit real consistency witness is not native IEEE. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem first_loop_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (t : Nat) (u : State α) (hu : FirstReady p u) :
    ∃ v, range A "i" t p.T 1
      (Program.backwardFirst Bool.false .fp32 .fp32 p r hasSeed).body u = some v ∧ FirstReady p v := by
  by_cases ht : t < p.T
  · obtain ⟨v,he,hv⟩ := first_body_success A p r hasSeed u t hu
    obtain ⟨w,hw,hr⟩ := first_loop_success A p r hasSeed (t+1) v hv
    refine ⟨w,?_,hr⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht,he,bind,Option.bind_some]
    exact hw
  · refine ⟨u,?_,hu⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht]
termination_by p.T-t
decreasing_by omega

theorem second_loop_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (t : Nat) (u : State α) (hu : SecondReady p u) :
    ∃ v, range A "_t" t p.T 1
      (Program.backwardSecond Bool.false .fp32 .fp32 p r hasSeed).body u = some v ∧ SecondReady p v := by
  by_cases ht : t < p.T
  · obtain ⟨v,he,hv⟩ := second_body_success A p r hasSeed u t hu
    obtain ⟨w,hw,hr⟩ := second_loop_success A p r hasSeed (t+1) v hv
    refine ⟨w,?_,hr⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht,he,bind,Option.bind_some]
    exact hw
  · refine ⟨u,?_,hu⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht]
termination_by p.T-t
decreasing_by omega

theorem range_statement {α : Type} [Inhabited α] (A : Algebra α)
    (name : RegName) (t stop : Nat) (body : List ComputeStmt) (u : State α) :
    step A (.forRange name t stop 1 body) u = range A name t stop 1 body u := by
  rw [step]

theorem exec_of_split {α : Type} [Inhabited α] (A : Algebra α) (kernel : ComputeKernel)
    (before between : List ComputeStmt) (first second : ComputeStmt)
    (hsplit : kernel.surfaceBody = before ++ [first] ++ between ++ [second])
    (u entry mid reverseEntry final : State α)
    (he : run A before u = some entry) (hfirst : step A first entry = some mid)
    (hm : run A between mid = some reverseEntry) (hsecond : step A second reverseEntry = some final) :
    FP.Structural.exec A kernel u = some final := by
  have hf : run A [first] entry = some mid := by
    simp only [run,hfirst,bind,Option.bind_some]
  have hs : run A [second] reverseEntry = some final := by
    simp only [run,hsecond,bind,Option.bind_some]
  change run A kernel.surfaceBody u = some final
  rw [hsplit,run_append,run_append,run_append,he]
  simp only [Option.bind_some,hf,hm,hs]

theorem original_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (u : State α) :
    ∃ v, FP.Structural.exec A (Program.backward Bool.false .fp32 .fp32 p r hasSeed) u = some v := by
  obtain ⟨entry,he,hr⟩ := prefix_success A p r hasSeed u
  obtain ⟨mid,hm,hready⟩ := first_loop_success A p r hasSeed 0 entry hr
  obtain ⟨reverseEntry,hrun,hreverse⟩ := middle_success A p r hasSeed mid hready
  obtain ⟨final,hfinal,hfinalReady⟩ := second_loop_success A p r hasSeed 0 reverseEntry hreverse
  refine ⟨final,?_⟩
  exact exec_of_split A _ _ _ _ _ (Program.backward_split Bool.false .fp32 .fp32 p r hasSeed)
    u entry mid reverseEntry final he ((range_statement A "i" 0 p.T _ entry).trans hm) hrun
    ((range_statement A "_t" 0 p.T _ reverseEntry).trans hfinal)

theorem nonempty_pair {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (u : State α) :
    ∃ v, FP.Structural.exec A (Program.backward Bool.false .fp32 .fp32 p r hasSeed) u = some v ∧
      FP.Structural.exec A (Program.backward Bool.true .fp32 .fp32 p r hasSeed) u = some v := by
  obtain ⟨v,hv⟩ := original_success A p r hasSeed u
  refine ⟨v,hv,?_⟩
  exact (FPProof.backward_whole_state .fp32 .fp32 p r hasSeed A u).symm.trans hv

def witnessRegions : BackwardRegions := ⟨"q","k","v","do","dq","dk","dv","seed"⟩

theorem witness_nonempty (T : Nat) (hasSeed : Bool) :
    ∃ v, FP.Structural.exec (engine TypedExecution.model)
      (Program.backward Bool.false .fp32 .fp32 (TypedExecution.witnessLayout T) witnessRegions hasSeed)
      TypedExecution.witnessInitial = some v ∧
      FP.Structural.exec (engine TypedExecution.model)
      (Program.backward Bool.true .fp32 .fp32 (TypedExecution.witnessLayout T) witnessRegions hasSeed)
      TypedExecution.witnessInitial = some v :=
  nonempty_pair (engine TypedExecution.model) (TypedExecution.witnessLayout T) witnessRegions hasSeed
    TypedExecution.witnessInitial

#axiomsClean first_loop_success
#axiomsClean second_loop_success
#axiomsClean range_statement
#axiomsClean exec_of_split
#axiomsClean original_success
#axiomsClean nonempty_pair
#axiomsClean witness_nonempty
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
