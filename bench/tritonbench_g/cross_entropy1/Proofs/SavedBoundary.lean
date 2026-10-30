import bench.tritonbench_g.cross_entropy1.Proofs.SavedMetadata

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Between calls, only the four captured saved allocations are protected.
Unrelated caller allocations may change, and new views/allocations may be
created. The current upstream is always read from the later world. -/
structure SavedPreserves {α σ : Type} (obs : Observations α σ) (c : Context)
    (before after : World α σ) : Prop where
  live : ∀ r, before.live r → after.live r
  memory : ∀ r i, (r=c.logits.region ∨ r=c.lse.region ∨ r=c.labels.region ∨ r=c.inplaceLogits.region) →
    after.memory r i=before.memory r i
  integers : ∀ i, obs.integer after c.labels i=obs.integer before c.labels i

theorem SavedPreserves.floatValue {α σ : Type} [Inhabited α] {obs : Observations α σ} {c : Context}
    {a b : World α σ} (between : SavedPreserves obs c a b) (t : Tensor)
    (saved : t.region=c.logits.region ∨ t.region=c.lse.region ∨ t.region=c.labels.region ∨ t.region=c.inplaceLogits.region)
    (i : Nat) : value obs b t i=value obs a t i := by
  unfold VeriTile.Bench.Repairs.CrossEntropy1.Host.value
  rw [between.memory t.region (t.index i) saved]

theorem SavedPreserves.context {α σ : Type} {obs : Observations α σ} {rt : Runtime α σ}
    {objects : ObjectRegistry rt} {c : Context} {a b : World α σ}
    (between : SavedPreserves obs c a b) (ctx : BackwardContext rt objects c a) : BackwardContext rt objects c b :=
  ⟨between.live _ ctx.logitsLive,between.live _ ctx.labelsLive,between.live _ ctx.lseLive,between.live _ ctx.destinationLive,
    ctx.logitsRank,ctx.logitsNotNeg,ctx.logitsObject,ctx.destinationObject,ctx.destinationShape,
    ctx.logitsLabels,ctx.logitsLSE,ctx.rows⟩

theorem SavedPreserves.lse {α σ : Type} [Inhabited α] {obs : Observations α σ} {c : Context}
    {a b : World α σ} (between : SavedPreserves obs c a b) {shape : List Nat} {values : Nat → α}
    (array : FloatArray obs a c.lse shape values) : FloatArray obs b c.lse shape values :=
  ⟨array.shapeEq,array.dtype,between.live _ array.live,
    fun i hi => (between.floatValue c.lse (Or.inr (Or.inl rfl)) i).trans (array.values i hi)⟩

theorem SavedPreserves.labels {α σ : Type} {obs : Observations α σ} {c : Context}
    {a b : World α σ} (between : SavedPreserves obs c a b) {shape : List Nat} {values : Nat → Int}
    (array : IntArray obs a c.labels shape values) : IntArray obs b c.labels shape values :=
  ⟨array.shapeEq,array.dtype,between.live _ array.live,fun i hi => (between.integers i).trans (array.values i hi)⟩

#axiomsClean SavedPreserves.floatValue
#axiomsClean SavedPreserves.context
#axiomsClean SavedPreserves.lse
#axiomsClean SavedPreserves.labels
#specNonCircular SavedPreserves avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
