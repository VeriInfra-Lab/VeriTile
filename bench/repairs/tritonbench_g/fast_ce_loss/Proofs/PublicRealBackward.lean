import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.PublicBackwardIO
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.RealIOValues

namespace VeriTile.Bench.Repairs.FastCeLoss.RealIO
open VeriTile.Triton FP.Structural KernelSource PublicIO
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

theorem backward_correct (p : Parameters) (row chunk : Nat) (flags : KernelReal.Flags p) :
    (backwardIO KernelReal.algebra some decode p row chunk Bool.true).Implements
      (RealIO.backwardFormula p chunk) := by
  intro xs _ bounds bounded initial binding
  let lifted := backwardValues p xs
  have hb : (backwardIO KernelReal.algebra id id p row chunk Bool.true).InBounds bounds lifted := by
    refine ⟨bounded.1,?_,?_,?_⟩
    · intro port lane active
      fin_cases port <;> exact bounded.2.1 _ lane active
    · intro port lane active
      fin_cases port <;> exact bounded.2.2.1 _ lane active
    · intro k; exact k.elim0
  have hi : (backwardIO KernelReal.algebra id id p row chunk Bool.true).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  obtain ⟨safe,final,ran,typed,frame⟩ := backward_implements KernelReal.algebra p row chunk Bool.true
    lifted trivial bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · intro r offset outside
    apply frame r offset
    refine ⟨?_,?_⟩
    · intro port lane active
      fin_cases port <;> exact outside.1 _ lane active
    · intro k; exact k.elim0
  let start := (backwardInvocation KernelReal.algebra p row chunk).initial initial
  let x := vector p (xs (1 : Fin 4))
  let saved := xs (2 : Fin 4) PUnit.unit
  let upstream := xs (3 : Fin 4) PUnit.unit
  have hx : ∀ col<p.vocab,
      (start.mem p.logits (start.pids 0*p.logitsStride+col)).read .real=some (x col) := by
    intro col hc
    simpa [start,x,vector,hc,backwardIO,backwardRead,column,width,mapValue] using
      binding (1 : Fin 4) (⟨col,hc⟩,PUnit.unit) (by simpa [backwardIO,backwardInputMask,column] using hc)
  have ht : label p start=backwardLabel p Bool.true xs :=
    backward_label KernelReal.algebra p row chunk Bool.true lifted initial hi
  have hl : (start.mem p.lse (start.pids 0)).read .real=some saved :=
    binding (2 : Fin 4) PUnit.unit trivial
  have hu : label p start≠ -100 →
      (start.mem p.upstream (start.pids 0*p.upstreamStride)).read .real=some upstream := by
    intro active
    exact binding (3 : Fin 4) PUnit.unit (by simpa only [ht] using active)
  obtain ⟨other,otherRan,values,_⟩ := KernelExecution.backward_outputs KernelReal.algebra p start
  have eq : other=final := Option.some.inj (otherRan.symm.trans (by
    simpa only [backwardIO,runSchedule_singleton,Invocation.run,backwardInvocation] using ran))
  subst other
  intro port lane active
  fin_cases port
  have cell := typed (0 : Fin 1) lane active
  have computed := values lane.1 active
  have formula : KernelModel.gradientValue KernelReal.algebra p start lane.1=
      some (KernelReal.gradient p (label p start) (start.pids 1*p.block+lane.1.val)
        (x (start.pids 1*p.block+lane.1.val)) saved upstream) := by
    by_cases ignored : label p start = -100
    · simp [KernelModel.gradientValue,KernelReal.gradient,ignored,KernelModel.cast,
        KernelModel.literal,KernelReal.algebra,RealProof.arithmetic]
    · exact KernelReal.gradient_finite p start flags lane.1 _ _ _ active (hx _ active) hl (hu ignored)
  rw [formula] at computed
  simpa only [ht] using real_cell _ _ _ cell computed

#axiomsClean backward_correct
end
end VeriTile.Bench.Repairs.FastCeLoss.RealIO
