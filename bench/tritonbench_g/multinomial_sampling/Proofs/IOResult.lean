import bench.tritonbench_g.multinomial_sampling.Proofs.LogicalTransport
import bench.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOResult
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem outcome (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
    (vals : (io rest P p pid).Inputs) (domain : Interface.SeedDomain rest P p pid vals)
    (s : State α) (bound : (io rest P p pid).InputBinding vals s) :
    ∃ t,runSchedule ((io rest P p pid).schedule vals.memory) s=some t ∧
      (io rest P p pid).OutputBinding vals (nativeOutputs rest P p pid vals) t ∧
      (io rest P p pid).Frame vals s t := by
  let entry := (job rest P p pid).initial s
  have hp : entry.pids 0=pid := rfl
  have boundEntry := IOFacts.entry_binding rest P p pid vals s bound
  have seeds := (IOFacts.seed_domain_iff rest P p pid vals entry boundEntry hp).mpr domain
  obtain ⟨t,ran,mem,values,frame⟩ := correctness rest P .fp32 p entry seeds
  refine ⟨t,?_,?_,?_⟩
  · change runSchedule [job rest P p pid] s=some t
    rw [runSchedule_singleton]
    exact ran
  · intro o i active
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change mapCell id (t.mem p.output (pid*p.block+i.1.val))=
      Cell.mk .int ((Logical.rounds P p pid (data rest P p pid vals) (count p)).output i)
    have ha : SourceValue.active p entry i := by
      simpa only [SourceValue.active,SourceValue.row,hp] using active
    have hv := values i ha
    simp only [SourceValue.row,hp] at hv
    rw [hv,LogicalTransport.rounds_eq rest P p pid vals entry boundEntry hp]
    rfl
  · intro r off outside
    have excluded : r≠p.output ∨ ∀ i : Row p,SourceValue.active p entry i → off≠SourceValue.row p entry i := by
      by_cases different : r≠p.output
      · exact Or.inl different
      · refine Or.inr ?_
        intro i active
        have ha : pid*p.block+i.1.val<p.batches := by
          simpa only [SourceValue.active,SourceValue.row,hp] using active
        have h := outside.1 (0 : Fin 1) i ha
        change r≠p.output ∨ off≠pid*p.block+i.1.val at h
        simpa only [SourceValue.row,hp] using h.resolve_left different
    simpa only [entry,Invocation.initial_mem] using frame r off excluded

#axiomsClean outcome
end
end VeriTile.Bench.Repairs.MultinomialSampling.IOResult
