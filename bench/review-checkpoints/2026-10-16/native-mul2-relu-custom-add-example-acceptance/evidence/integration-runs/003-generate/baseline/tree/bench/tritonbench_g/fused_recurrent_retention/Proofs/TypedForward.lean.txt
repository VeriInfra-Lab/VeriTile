import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBody
import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedFinal

/-! Complete successful typed source/candidate execution follows the actual
structural interpreter. Every shape, T=0, both feature flags and arbitrary
initial registers/memory are covered for the selected fp32 storage profile.
No numerical backend law is used; native/IEEE refinement remains separate. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem loop_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (t : Nat) (u : State α) (hu : Ready p u) :
    ∃ v, range A "_t" t p.T 1
      (Program.forwardParts Bool.false .fp32 .fp32 p r hasSeed storeFinal).body u = some v ∧ Ready p v := by
  by_cases ht : t < p.T
  · obtain ⟨v,he,hv⟩ := body_success A p r hasSeed storeFinal u t hu
    obtain ⟨w,hw,hr⟩ := loop_success A p r hasSeed storeFinal (t+1) v hv
    refine ⟨w,?_,hr⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht,he,bind,Option.bind_some]
    exact hw
  · refine ⟨u,?_,hu⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht]
termination_by p.T-t
decreasing_by omega

theorem statement_eval {α : Type} [Inhabited α] (A : Algebra α)
    (t stop : Nat) (body : List ComputeStmt) (u : State α) :
    step A (.forRange "_t" t stop 1 body) u = range A "_t" t stop 1 body u := by
  rw [step]

theorem exec_of_split {α : Type} [Inhabited α] (A : Algebra α)
    (kernel : ComputeKernel) (before : List ComputeStmt) (middle : ComputeStmt) (after : List ComputeStmt)
    (hsplit : kernel.surfaceBody = before ++ [middle] ++ after) (u entry mid final : State α)
    (he : run A before u = some entry) (hm : step A middle entry = some mid)
    (hf : run A after mid = some final) : FP.Structural.exec A kernel u = some final := by
  have hsingle : run A [middle] entry = some mid := by
    simp only [run,hm,bind,Option.bind_some]
  change run A kernel.surfaceBody u = some final
  rw [hsplit,run_append,run_append,he]
  simp only [Option.bind_some,hsingle,hf]

theorem original_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) :
    ∃ v, FP.Structural.exec A (Program.forward Bool.false .fp32 .fp32 p r hasSeed storeFinal) u =
      some v := by
  obtain ⟨entry,he,hr⟩ := prefix_success A p r hasSeed storeFinal u
  obtain ⟨mid,hm,hready⟩ := loop_success A p r hasSeed storeFinal 0 entry hr
  obtain ⟨final,hf⟩ := final_success A p r hasSeed storeFinal mid hready
  refine ⟨final,?_⟩
  exact exec_of_split A _ _ _ _ (Program.forward_split Bool.false .fp32 .fp32 p r hasSeed storeFinal)
    u entry mid final he ((statement_eval A 0 p.T _ entry).trans hm) hf

theorem nonempty_pair {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) :
    ∃ v, FP.Structural.exec A (Program.forward Bool.false .fp32 .fp32 p r hasSeed storeFinal) u =
      some v ∧ FP.Structural.exec A
        (Program.forward Bool.true .fp32 .fp32 p r hasSeed storeFinal) u = some v := by
  obtain ⟨v,hv⟩ := original_success A p r hasSeed storeFinal u
  refine ⟨v,hv,?_⟩
  exact (FPProof.forward_whole_state .fp32 .fp32 p r hasSeed storeFinal A u).symm.trans hv


/-- A nonzero launch with both key and value tails, starting with empty
registers. The witness backend is explicitly mathematical, not native IEEE. -/
def witnessLayout (T : Nat) : Layout :=
  ⟨9,3,1,15,5,1,2,2,T,2,4,3,5,1⟩

def witnessRegions : ForwardRegions := ⟨"q","k","v","o","seed","final"⟩

def witnessInitial : State (WithBot ℝ) where
  mem := fun _ _ => .mk .real (some 1)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

theorem witness_nonempty (T : Nat) (hasSeed storeFinal : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (Program.forward Bool.false .fp32 .fp32 (witnessLayout T) witnessRegions hasSeed storeFinal)
      witnessInitial = some v ∧
      FP.Structural.exec (engine model)
      (Program.forward Bool.true .fp32 .fp32 (witnessLayout T) witnessRegions hasSeed storeFinal)
      witnessInitial = some v :=
  nonempty_pair (engine model) (witnessLayout T) witnessRegions hasSeed storeFinal witnessInitial

#axiomsClean loop_success
#axiomsClean statement_eval
#axiomsClean exec_of_split
#axiomsClean original_success
#axiomsClean nonempty_pair
#axiomsClean witness_nonempty
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
