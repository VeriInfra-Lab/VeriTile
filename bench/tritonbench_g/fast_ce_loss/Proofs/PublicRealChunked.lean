import bench.tritonbench_g.fast_ce_loss.Proofs.PublicChunkedIO
import bench.tritonbench_g.fast_ce_loss.Proofs.RealIOValues

namespace VeriTile.Bench.Repairs.FastCeLoss.RealIO
open VeriTile.Triton FP.Structural KernelSource PublicIO
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

theorem chunked_correct (p : Parameters) (row chunk : Nat) (flags : KernelReal.Flags p)
    (positive : 0<p.block) (nonempty : chunk*p.block<p.vocab) (distinct : p.loss≠p.lse) :
    (forwardIO KernelReal.algebra some decode p row chunk Bool.true Bool.true).ImplementsOn
      (domain p) (RealIO.chunkedFormula p chunk) := by
  intro xs hd bounds bounded initial binding
  let lifted := forwardValues p xs
  have hb : (forwardIO KernelReal.algebra id id p row chunk Bool.true Bool.true).InBounds bounds lifted := by
    refine ⟨bounded.1,?_,?_,?_⟩
    · intro port lane active
      fin_cases port <;> exact bounded.2.1 _ lane active
    · intro port lane active
      fin_cases port <;> exact bounded.2.2.1 _ lane active
    · intro k; exact k.elim0
  have hi : (forwardIO KernelReal.algebra id id p row chunk Bool.true Bool.true).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  have valid : nativeDomain p Bool.true lifted := by
    exact hd.imp_right And.left
  obtain ⟨safe,final,ran,typed,frame⟩ := chunked_implements KernelReal.algebra p row chunk Bool.true
    positive distinct lifted valid bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · intro r offset outside
    apply frame r offset
    refine ⟨?_,?_⟩
    · intro port lane active
      fin_cases port <;> exact outside.1 _ lane active
    · intro k; exact k.elim0
  let start := (forwardInvocation KernelReal.algebra p row chunk Bool.true).initial initial
  let x := vector p (xs (1 : Fin 3))
  have hx : ∀ col<p.vocab,
      (start.mem p.logits (start.pids 0*p.logitsStride+col)).read .real=some (x col) := by
    intro col hc
    simpa [start,x,vector,hc,forwardIO,forwardRead,column,width,mapValue] using
      binding (1 : Fin 3) (⟨col,hc⟩,PUnit.unit) (by simpa [forwardIO,forwardInputMask,column] using hc)
  have ht : label p start=forwardLabel p Bool.true xs :=
    forward_label KernelReal.algebra p row chunk Bool.true Bool.true lifted initial hi
  have domain' : label p start = -100 ∨ (0≤label p start ∧ label p start<(p.vocab : Int)) := by
    simpa only [ht,domain] using hd
  obtain ⟨other,otherRan,lse,loss,_⟩ := KernelInterface.Real.chunked_kernel p start flags x
    positive nonempty hx domain' distinct
  have eq : other=final := Option.some.inj (otherRan.symm.trans (by
    simpa only [forwardIO,runSchedule_singleton,Invocation.run,forwardInvocation,if_true] using ran))
  subst other
  intro port lane active
  fin_cases port <;> cases lane
  · have cell := typed (0 : Fin 2) PUnit.unit active
    exact real_cell _ _ _ cell lse
  · have cell := typed (1 : Fin 2) PUnit.unit active
    have first : chunk=0 := active.resolve_left (by decide)
    simpa only [ht] using real_cell _ _ _ cell (loss first)

#axiomsClean chunked_correct
end
end VeriTile.Bench.Repairs.FastCeLoss.RealIO
