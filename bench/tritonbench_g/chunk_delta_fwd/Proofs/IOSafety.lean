import bench.tritonbench_g.chunk_delta_fwd.IO
import bench.tritonbench_g.chunk_delta_fwd.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.IOSafety
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
noncomputable section
set_option maxHeartbeats 8000000
variable {α β : Type} [Inhabited β]

theorem live (p : Parameters) (shape : Geometry.ShapeValid p) (pid : PID)
    (hp : Geometry.ValidPID p pid) : 0 < p.V := by
  by_contra hn
  have hz : p.V = 0 := by omega
  have hc : ceilDiv p.V p.BV = 0 := by
    rw [hz]
    apply Nat.div_eq_of_lt
    have := shape.bv_pos
    omega
  have hv := hp.1
  rw [hc] at hv
  omega

theorem bounds (M : Algebra β) (D : DotPrimitive β) (p : Parameters)
    (shape : Geometry.ShapeValid p) (schedule : List PID) (encode : α → β) (observe : β → α)
    (xs : Inputs α p) (bounds : RegionName → Nat)
    (hb : (io M D p schedule encode observe).InBounds bounds xs)
    (pid : PID) (hp : Geometry.ValidPID p pid) : SourceSafety.Bounds p pid bounds := by
  have hv := live p shape pid hp
  constructor
  · intro t k ht hk
    exact hb.2.1 (0 : Fin 4) (⟨pid.2,hp.2⟩,⟨t,ht⟩,⟨k,hk⟩,PUnit.unit) hv
  · intro t k ht hk
    exact hb.2.1 (1 : Fin 4) (⟨pid.2,hp.2⟩,⟨t,ht⟩,⟨k,hk⟩,PUnit.unit) hv
  · intro t v ht hv
    exact hb.2.1 (2 : Fin 4) (⟨pid.2,hp.2⟩,⟨t,ht⟩,⟨v,hv⟩,PUnit.unit) trivial
  · intro on k v hk hv
    exact hb.2.1 (3 : Fin 4) (⟨pid.2,hp.2⟩,⟨k,hk⟩,⟨v,hv⟩,PUnit.unit) on
  · intro c k v hc hk hv
    have h := hb.2.2.1 (0 : Fin 3) (⟨pid.2,hp.2⟩,⟨c,hc⟩,⟨k,hk⟩,⟨v,hv⟩,PUnit.unit) trivial
    change ((pid.2*p.NT+c)*p.K+k)*p.V+v < bounds "history" at h
    convert h using 1 <;> ring
  · intro t v ht hv
    have h := hb.2.2.1 (1 : Fin 3) (⟨pid.2,hp.2⟩,⟨t,ht⟩,⟨v,hv⟩,PUnit.unit) trivial
    change (pid.2*p.T+t)*p.V+v < bounds "corrected" at h
    convert h using 1 <;> ring
  · intro on k v hk hv
    have h := hb.2.2.1 (2 : Fin 3) (⟨pid.2,hp.2⟩,⟨k,hk⟩,⟨v,hv⟩,PUnit.unit) on
    change (pid.2*p.K+k)*p.V+v < bounds "final" at h
    convert h using 1 <;> ring

/-- Every invocation is checked in the memory produced by its actual
predecessor. Geometry proves source access bounds independently of payloads. -/
theorem schedule (M : Algebra β) (D : DotPrimitive β) (p : Parameters)
    (shape : Geometry.ShapeValid p) (work : List PID) (order : work.Perm (grid p))
    (encode : α → β) (observe : β → α) (xs : Inputs α p) (sizes : RegionName → Nat)
    (hb : (io M D p work encode observe).InBounds sizes xs) (s : State β) :
    ScheduleSafe sizes ((io M D p work encode observe).schedule xs) s := by
  have allBounds : ∀ pid, Geometry.ValidPID p pid → SourceSafety.Bounds p pid sizes :=
    bounds M D p shape work encode observe xs sizes hb
  have go : ∀ jobs : List PID, (∀ pid ∈ jobs, Geometry.ValidPID p pid) → ∀ start : State β,
      ScheduleSafe sizes (jobs.map (invocation M D p)) start := by
    intro jobs valid start
    induction jobs generalizing start with
    | nil => trivial
    | cons pid rest ih =>
      refine ⟨?_,fun t _ => ih (fun other ho => valid other (by simp [ho])) t⟩
      exact SourceSafety.source M D p ((invocation M D p pid).initial start) rfl sizes
        (allBounds pid (valid pid (by simp)))
  exact go work (fun pid hp => (mem_grid p pid).mp (order.mem_iff.mp hp)) s

#axiomsClean schedule
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.IOSafety
