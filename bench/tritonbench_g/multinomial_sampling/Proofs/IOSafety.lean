import bench.tritonbench_g.multinomial_sampling.Proofs.SourceSafety
import bench.tritonbench_g.multinomial_sampling.Proofs.IOFacts

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOSafety
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem access (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
    (vals : (io rest P p pid).Inputs) (bounds : RegionName → Nat)
    (backed : (io rest P p pid).InBounds bounds vals) (s : State α)
    (samePid : s.pids 0=pid) : SourceSafety.Access p s bounds := by
  constructor
  · intro i active
    have hr : pid*p.block+i.1.val<p.batches := by simpa only [SourceValue.active,SourceValue.row,samePid] using active
    simpa only [SourceValue.row,samePid] using backed.2.1 (1 : Fin 4) (⟨_,hr⟩,()) trivial
  · intro i active
    have hr : pid*p.block+i.1.val<p.batches := by simpa only [SourceValue.active,SourceValue.row,samePid] using active
    simpa only [SourceValue.row,samePid] using backed.2.1 (2 : Fin 4) (⟨_,hr⟩,()) trivial
  · intro i active j hj
    have hr : pid*p.block+i.1.val<p.batches := by simpa only [SourceValue.active,SourceValue.row,samePid] using active
    simpa only [SourceValue.row,samePid] using backed.2.1 (0 : Fin 4) (⟨_,hr⟩,⟨j,hj⟩,()) trivial
  · intro i active j hj
    have hr : pid*p.block+i.1.val<p.batches := by simpa only [SourceValue.active,SourceValue.row,samePid] using active
    simpa only [SourceValue.row,samePid] using backed.2.1 (3 : Fin 4) (⟨_,hr⟩,⟨j,hj⟩,()) trivial
  · intro i active
    have hr : pid*p.block+i.1.val<p.batches := by simpa only [SourceValue.active,SourceValue.row,samePid] using active
    simpa only [SourceValue.row,samePid] using backed.2.2.1 (0 : Fin 1) i hr

theorem whole (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
    (vals : (io rest P p pid).Inputs) (domain : Interface.SeedDomain rest P p pid vals)
    (bounds : RegionName → Nat) (backed : (io rest P p pid).InBounds bounds vals)
    (s : State α) (bound : (io rest P p pid).InputBinding vals s) :
    ScheduleSafe bounds ((io rest P p pid).schedule vals.memory) s := by
  change ScheduleSafe bounds [job rest P p pid] s
  rw [scheduleSafe_singleton]
  let entry := (job rest P p pid).initial s
  have boundEntry := IOFacts.entry_binding rest P p pid vals s bound
  have hp : entry.pids 0=pid := rfl
  have seeds := (IOFacts.seed_domain_iff rest P p pid vals entry boundEntry hp).mpr domain
  exact SourceSafety.whole rest P p entry seeds bounds (access rest P p pid vals bounds backed entry hp)

#axiomsClean access
#axiomsClean whole
end
end VeriTile.Bench.Repairs.MultinomialSampling.IOSafety
