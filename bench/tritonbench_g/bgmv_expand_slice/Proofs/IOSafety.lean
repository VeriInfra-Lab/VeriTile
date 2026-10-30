import bench.tritonbench_g.bgmv_expand_slice.Proofs.SourceSafety
import bench.tritonbench_g.bgmv_expand_slice.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem invocation_safe (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (valid : Valid xs) (bounds : RegionName → Nat)
    (hb : (sourceIO M R project h).InBounds bounds xs)
    (p : Geometry.Program h) (s : State A) (metadata : IndexBinding h xs s) :
    (invocation M R h p).Safe bounds s := by
  have positive : 0<h.N := by have := p.rowInside;omega
  let batch : Fin h.batches := ⟨p.batch,p.batchInside⟩
  have hi := metadata positive batch
  have domain : (PublicExecution.data s).indices p.batch=-1 ∨
      0≤(PublicExecution.data s).indices p.batch := by rw [hi];exact valid batch
  apply PublicExecution.source_safe M R h p s domain bounds
  · exact hb.2.1 ⟨3,by simp [sourceIO,inputs]⟩ (batch,PUnit.unit) positive
  · intro active j access
    have hj : j.val < inputWidth h := by
      unfold inputWidth
      cases he : h.even
      · simp only [he,Bool.false_eq_true,↓reduceIte]
        exact access.resolve_left (by simp [he])
      · simp only [he,↓reduceIte]
        exact j.isLt
    exact hb.2.1 ⟨0,by simp [sourceIO,inputs]⟩ (batch,⟨j.val,hj⟩,PUnit.unit)
      ⟨positive,by rwa [hi] at active⟩
  · intro active j hj
    have bound := hb.2.1 ⟨1,by simp [sourceIO,inputs]⟩
      (batch,⟨p.row,p.rowInside⟩,⟨j.val,hj⟩,PUnit.unit) (by rwa [hi] at active)
    change (index xs batch).toNat*(h.N*h.K)+p.row*h.K+j.val<bounds "W" at bound
    rw [hi]
    exact bound
  · intro active
    exact hb.2.2.1 ⟨0,by simp [sourceIO]⟩ (batch,⟨p.row,p.rowInside⟩,PUnit.unit)
      (by rwa [hi] at active)

theorem schedule_safe (M : Algebra A) (R : Reduction A) (project : A → A)
    (h : Call) (xs : Inputs A h) (valid : Valid xs) (bounds : RegionName → Nat)
    (hb : (sourceIO M R project h).InBounds bounds xs)
    (jobs : List (Geometry.Program h)) (s : State A) (metadata : IndexBinding h xs s) :
    ScheduleSafe bounds (jobs.map (invocation M R h)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons p jobs ih =>
    refine ⟨invocation_safe M R project h xs valid bounds hb p s metadata,?_⟩
    intro t ran
    have positive : 0<h.N := by have := p.rowInside;omega
    have domain := valid ⟨p.batch,p.batchInside⟩
    rw [←metadata positive ⟨p.batch,p.batchInside⟩] at domain
    obtain ⟨u,ran',mem⟩ := PublicExecution.source_executes M R h p s domain
    have same : t=u := Option.some.inj (ran.symm.trans ran')
    subst t
    apply ih
    intro hn batch
    rw [PublicExecution.step_indices M R h p s u mem]
    exact metadata hn batch

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
