import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedSteps

/-! Complete repaired quotation execution in the shared Structural interpreter.
The scale register is an external launch argument. No successful execution,
precomputed carry, numerical result or whole-kernel relation is a premise. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 16000

private theorem range_success {α : Type} [Inhabited α] (A : Algebra α)
    (name : RegName) (body : List ComputeStmt) (Ready : State α → Prop)
    (bodyStep : ∀ u t, Ready u → ∃ v,
      run A body (u.setReg name .nat [] (fun _ => t)) = some v ∧ Ready v)
    (start stop : Nat) (u : State α) (hu : Ready u) :
    ∃ v, range A name start stop 1 body u = some v ∧ Ready v := by
  by_cases ht : start < stop
  · obtain ⟨v,he,hv⟩ := bodyStep u start hu
    obtain ⟨w,hw,hr⟩ := range_success A name body Ready bodyStep (start+1) stop v hv
    refine ⟨w,?_,hr⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht,he,bind,Option.bind_some]
    exact hw
  · refine ⟨u,?_,hu⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,ht]
termination_by stop-start
decreasing_by omega

private theorem run_single_loop {α : Type} [Inhabited α] (A : Algebra α)
    (name : RegName) (stop : Nat) (body : List ComputeStmt) (u : State α) :
    run A [.forRange name 0 stop 1 body] u = range A name 0 stop 1 body u := by
  simp only [run,step,bind]
  cases range A name 0 stop 1 body u <;> rfl

theorem forward_execution {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) (u : State α)
    (scale : Values α .real []) (hscale : u.regs .real [] "scale" = some scale) :
    ∃ v, FP.Structural.exec A (Program.forward p r hasSeed storeFinal) u = some v := by
  obtain ⟨entry,he,hr⟩ := forward_prefix A p r hasSeed storeFinal u scale hscale
  obtain ⟨mid,hm,hready⟩ := range_success A "_t" _ (ForwardReady p)
    (forward_body A p r hasSeed storeFinal) 0 p.T entry hr
  obtain ⟨final,hf⟩ := forward_final A p r hasSeed storeFinal mid hready
  refine ⟨final,?_⟩
  change run A (Program.forward p r hasSeed storeFinal).surfaceBody u = some final
  rw [Program.forward_split,run_append,run_append,he]
  simp only [Option.bind_some,run_single_loop,hm,hf]

theorem backward_execution {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed hasFinal : Bool) (u : State α)
    (scale : Values α .real []) (hscale : u.regs .real [] "scale" = some scale) :
    ∃ v, FP.Structural.exec A (Program.backward p r hasSeed hasFinal) u = some v := by
  obtain ⟨entry,he,hr⟩ := backward_prefix A p r hasSeed hasFinal u scale hscale
  obtain ⟨mid,hm,hready⟩ := range_success A "i" _ (FirstReady p)
    (backward_first A p r hasSeed hasFinal) 0 p.T entry hr
  obtain ⟨reverseEntry,hre,hrr⟩ := backward_middle A p r hasSeed hasFinal mid hready
  obtain ⟨final,hf,hreadyFinal⟩ := range_success A "_t" _ (SecondReady p)
    (backward_second A p r hasSeed hasFinal) 0 p.T reverseEntry hrr
  refine ⟨final,?_⟩
  change run A (Program.backward p r hasSeed hasFinal).surfaceBody u = some final
  rw [Program.backward_split,run_append,run_append,run_append,he]
  simp only [Option.bind_some,run_single_loop,hm,hre,hf]

#axiomsClean forward_execution
#axiomsClean backward_execution
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
