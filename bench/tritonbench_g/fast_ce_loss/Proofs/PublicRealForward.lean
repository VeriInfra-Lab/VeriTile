import bench.tritonbench_g.fast_ce_loss.Proofs.PublicForwardIO
import bench.tritonbench_g.fast_ce_loss.Proofs.RealIOValues

namespace VeriTile.Bench.Repairs.FastCeLoss.RealIO
open VeriTile.Triton FP.Structural KernelSource PublicIO
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

theorem forward_correct (p : Parameters) (row : Nat) (flags : KernelReal.Flags p)
    (positive : 0<p.vocab) (covered : p.vocab≤p.block) (distinct : p.loss≠p.lse) :
    (forwardIO KernelReal.algebra some decode p row 0 Bool.false Bool.true).ImplementsOn
      (domain p) (RealIO.forwardFormula p) := by
  intro xs hd bounds bounded initial binding
  let lifted := forwardValues p xs
  have hb : (forwardIO KernelReal.algebra id id p row 0 Bool.false Bool.true).InBounds bounds lifted := by
    refine ⟨bounded.1,?_,?_,?_⟩
    · intro port lane active
      fin_cases port <;> exact bounded.2.1 _ lane active
    · intro port lane active
      fin_cases port <;> exact bounded.2.2.1 _ lane active
    · intro k; exact k.elim0
  have hi : (forwardIO KernelReal.algebra id id p row 0 Bool.false Bool.true).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  have valid : nativeDomain p Bool.true lifted := by
    exact hd.imp_right And.left
  obtain ⟨safe,final,ran,typed,frame⟩ := forward_implements KernelReal.algebra p row Bool.true
    (lt_of_lt_of_le positive covered) distinct lifted valid bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · intro r offset outside
    apply frame r offset
    refine ⟨?_,?_⟩
    · intro port lane active
      fin_cases port <;> exact outside.1 _ lane active
    · intro k; exact k.elim0
  let start := (forwardInvocation KernelReal.algebra p row 0 Bool.false).initial initial
  let x := vector p (xs (1 : Fin 3))
  have hx : ∀ col<p.vocab,
      (start.mem p.logits (start.pids 0*p.logitsStride+col)).read .real=some (x col) := by
    intro col hc
    simpa [start,x,vector,hc,forwardIO,forwardRead,column,width,mapValue] using
      binding (1 : Fin 3) (⟨col,hc⟩,PUnit.unit) (by simpa [forwardIO,forwardInputMask,column] using hc)
  have ht : label p start=forwardLabel p Bool.true xs :=
    forward_label KernelReal.algebra p row 0 Bool.false Bool.true lifted initial hi
  have domain' : label p start = -100 ∨ (0≤label p start ∧ label p start<(p.vocab : Int)) := by
    simpa only [ht,domain] using hd
  obtain ⟨other,otherRan,lse,loss,_⟩ := KernelInterface.Real.forward_kernel p start flags x
    positive covered hx domain' distinct
  have eq : other=final := Option.some.inj (otherRan.symm.trans (by
    simpa only [forwardIO,runSchedule_singleton,Invocation.run,forwardInvocation,if_false] using ran))
  subst other
  intro port lane active
  fin_cases port <;> cases lane
  · have cell := typed (0 : Fin 2) PUnit.unit active
    exact real_cell _ _ _ cell lse
  · have cell := typed (1 : Fin 2) PUnit.unit active
    simpa only [ht] using real_cell _ _ _ cell loss

#axiomsClean forward_correct
end
end VeriTile.Bench.Repairs.FastCeLoss.RealIO
