import bench.repairs.tritonbench_g.rotary_emb_nopad.PairIO
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.UncachedInterface
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.CachedInterface

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.PairIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem u_run (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) (s : State α) :
    (uInvocation rest ops g p q).run s = (RotationIO.uInvocation rest ops g p q).run s :=
  (AddressCSE.uncached_execution (Typed.algebra rest ops) (Typed.uLayout g.request)
    ((RotationIO.uInvocation rest ops g p q).initial s)).symm

theorem c_run (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) (s : State α) :
    (cInvocation rest ops g p q).run s = (RotationIO.cInvocation rest ops g p q).run s :=
  (AddressCSE.cached_execution (Typed.algebra rest ops) (Typed.cLayout g.request)
    ((RotationIO.cInvocation rest ops g p q).initial s)).symm

theorem u_safe (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (RotationIO.uInvocation rest ops g p q).Safe bounds s ↔
      (uInvocation rest ops g p q).Safe bounds s :=
  AddressCSE.uncached_safe (Typed.algebra rest ops) (Typed.uLayout g.request) bounds
    ((RotationIO.uInvocation rest ops g p q).initial s)

theorem c_safe (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (RotationIO.cInvocation rest ops g p q).Safe bounds s ↔
      (cInvocation rest ops g p q).Safe bounds s :=
  AddressCSE.cached_safe (Typed.algebra rest ops) (Typed.cLayout g.request) bounds
    ((RotationIO.cInvocation rest ops g p q).initial s)

/-- The exact connection composes over arbitrary actual launches and their
intermediate states, beyond the public arbitrary-single-tile IO scope. -/
theorem u_schedule_run (rest : Algebra α) (ops : Ops α) (g : Geometry)
    (jobs : List (Nat × Nat)) (s : State α) :
    runSchedule (jobs.map (fun pq => uInvocation rest ops g pq.1 pq.2)) s =
      runSchedule (jobs.map (fun pq => RotationIO.uInvocation rest ops g pq.1 pq.2)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pq jobs ih =>
    simp only [List.map_cons,runSchedule,u_run]
    exact congrArg (fun f => (RotationIO.uInvocation rest ops g pq.1 pq.2).run s >>= f) (funext ih)

theorem c_schedule_run (rest : Algebra α) (ops : Ops α) (g : Geometry)
    (jobs : List (Nat × Nat)) (s : State α) :
    runSchedule (jobs.map (fun pq => cInvocation rest ops g pq.1 pq.2)) s =
      runSchedule (jobs.map (fun pq => RotationIO.cInvocation rest ops g pq.1 pq.2)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pq jobs ih =>
    simp only [List.map_cons,runSchedule,c_run]
    exact congrArg (fun f => (RotationIO.cInvocation rest ops g pq.1 pq.2).run s >>= f) (funext ih)

theorem u_schedule_safe (rest : Algebra α) (ops : Ops α) (g : Geometry)
    (jobs : List (Nat × Nat)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (fun pq => RotationIO.uInvocation rest ops g pq.1 pq.2)) s ↔
      ScheduleSafe bounds (jobs.map (fun pq => uInvocation rest ops g pq.1 pq.2)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pq jobs ih =>
    simp only [List.map_cons,ScheduleSafe,u_run]
    exact and_congr (u_safe rest ops g pq.1 pq.2 bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

theorem c_schedule_safe (rest : Algebra α) (ops : Ops α) (g : Geometry)
    (jobs : List (Nat × Nat)) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (fun pq => RotationIO.cInvocation rest ops g pq.1 pq.2)) s ↔
      ScheduleSafe bounds (jobs.map (fun pq => cInvocation rest ops g pq.1 pq.2)) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pq jobs ih =>
    simp only [List.map_cons,ScheduleSafe,c_run]
    exact and_congr (c_safe rest ops g pq.1 pq.2 bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

theorem uncached_implements (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (heads : ∀ key,0<g.request.heads key) :
    (uncached rest ops g p q).toProgramIO.Implements (RotationIO.uTarget ops g) := by
  intro xs admitted bounds backed initial inputs
  obtain ⟨safe,final,ran,outputs,frame⟩ := RotationIO.uncached_implements rest ops g p q heads
    xs admitted bounds backed initial inputs
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (scheduleSafe_singleton _ _ _).mpr
      ((u_safe rest ops g p q bounds initial).mp ((scheduleSafe_singleton _ _ _).mp safe))
  · exact (runSchedule_singleton _ _).trans ((u_run rest ops g p q initial).trans
      ((runSchedule_singleton _ _).symm.trans ran))

theorem cached_implements (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (token : q<g.tokens) :
    (cached rest ops g p q).toProgramIO.ImplementsOn (RotationIO.MetadataDomain g)
      (RotationIO.cTarget ops g) := by
  intro xs admitted bounds backed initial inputs
  obtain ⟨safe,final,ran,outputs,frame⟩ := RotationIO.cached_implements rest ops g p q token
    xs admitted bounds backed initial inputs
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (scheduleSafe_singleton _ _ _).mpr
      ((c_safe rest ops g p q bounds initial).mp ((scheduleSafe_singleton _ _ _).mp safe))
  · exact (runSchedule_singleton _ _).trans ((c_run rest ops g p q initial).trans
      ((runSchedule_singleton _ _).symm.trans ran))

theorem uncached_equivalent (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (heads : ∀ key,0<g.request.heads key) :
    (RotationIO.uncachedIO rest ops g p q).Equivalent (uncached rest ops g p q) :=
  ProgramIO.EquivalentOn.of_implementsOn (RotationIO.uncached_implements rest ops g p q heads)
    (uncached_implements rest ops g p q heads)

theorem cached_equivalent (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat)
    (token : q<g.tokens) :
    (RotationIO.cachedIO rest ops g p q).EquivalentOn (cached rest ops g p q) (RotationIO.MetadataDomain g) :=
  ProgramIO.EquivalentOn.of_implementsOn (RotationIO.cached_implements rest ops g p q token)
    (cached_implements rest ops g p q token)

#axiomsClean uncached_equivalent
#axiomsClean cached_equivalent
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.PairIO
