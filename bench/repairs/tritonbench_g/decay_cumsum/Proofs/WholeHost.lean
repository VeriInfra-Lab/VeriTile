import bench.repairs.tritonbench_g.decay_cumsum.Proofs.HostPreparation

/-! Additive whole-domain entry. Allocation metadata is an output of successful
preparation. The error alternative is inhabited even when no placement can
exist for the requested dimensions. No capacity, freshness, snapshot or source
execution premise is required on a preparation/allocation-failure branch.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.WholeHost
open VeriTile.Triton PackedStorage LogicalStorage
noncomputable section
set_option maxHeartbeats 5000000
variable {β γ : Type}

structure Prepared (h : Call) (kind : Kind) (γ : Type) where
  storage : Placement h kind
  heap : Heap γ

structure API (h : Call) (kind : Kind) (β γ : Type) where
  prepare : Except String (Prepared h kind γ)
  setup : Prepared h kind γ → Except String Unit
  copy : RegionName → List (Copyback.Job β) → Copyback.Heap β → Except (Host.Failure β) (Copyback.Heap β)

/-- This adapter is created only after an actual successful preparation. -/
def preparedAPI {h : Call} {kind : Kind} (api : API h kind β γ) (p : Prepared h kind γ) : Host.API β γ :=
  ⟨.ok p.heap,api.setup p,api.copy⟩

structure PreparationLaws (h : Call) (kind : Kind) (api : API h kind β γ)
    (tensors : RegionName → Host.Target) (decode : Bool → (Fin 4 → β) → γ)
    (before : Copyback.Heap β) (live : Nat → Prop) : Prop where
  snapshots : ∀ p, api.prepare=.ok p →
    Snapshot h kind p.storage p.heap (Host.logicalData h tensors decode before)
  reshapes : ∀ p, api.prepare=.ok p → ∀ r∈Roles.inputs kind,
    (tensors r).shape.prod=Logical.cells h
  fresh : ∀ p, api.prepare=.ok p → ∀ r∈Roles.buffers kind, ∀ off, off<h.cells → ∀ byte : Fin 4,
    ¬live (address p.storage r off+byte.val)

structure CopyLaws {h : Call} {kind : Kind} (api : API h kind β γ) : Prop where
  success : ∀ r jobs before after, api.copy r jobs before=.ok after → after=Copyback.commit jobs before
  failure : ∀ r jobs before error, api.copy r jobs before=.error error → error.stage=.copy ∧
    ∀ addr, (∀ job∈jobs, ¬(job.target≤addr ∧ addr<job.target+4)) → error.heap addr=before addr

theorem copy_laws {h : Call} {kind : Kind} (api : API h kind β γ) (laws : CopyLaws api)
    (p : Prepared h kind γ) : Host.CopyLaws (preparedAPI api p) := ⟨laws.success,laws.failure⟩

def run (h : Call) (kind : Kind) (api : API h kind β γ) (tensors : RegionName → Host.Target)
    (encode : Bool → γ → Fin 4 → β)
    (grid : Prepared h kind γ → Heap γ → Option (Heap γ))
    (crop : Prepared h kind γ → Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β) : Except (Host.Failure β) (Host.Returned β) :=
  if Logical.cells h=0 then .ok ⟨(),tensors,before⟩ else
    match api.prepare with
    | .error message => .error ⟨.preparation,message,before⟩
    | .ok p => Host.run (preparedAPI api p) h kind tensors encode (grid p) (crop p) before

/-- No placement is even a parameter of the preparation-failure entry. Empty
calls still return before preparation, exactly as the Python wrapper does. -/
theorem preparation_failure (h : Call) (kind : Kind) (api : API h kind β γ)
    (tensors : RegionName → Host.Target) (encode : Bool → γ → Fin 4 → β)
    (grid : Prepared h kind γ → Heap γ → Option (Heap γ))
    (crop : Prepared h kind γ → Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β) (message : String) (failed : api.prepare=.error message) :
    run h kind api tensors encode grid crop before=
      if Logical.cells h=0 then .ok ⟨(),tensors,before⟩ else .error ⟨.preparation,message,before⟩ := by
  simp only [run,failed]

/-- The ordinary local preparation contract is vacuous on error without any
allocator-capacity/freshness witness, including an impossible requested size. -/
theorem preparation_laws_on_failure (h : Call) (kind : Kind) (api : API h kind β γ)
    (tensors : RegionName → Host.Target) (decode : Bool → (Fin 4 → β) → γ)
    (before : Copyback.Heap β) (live : Nat → Prop) (message : String) (failed : api.prepare=.error message) :
    PreparationLaws h kind api tensors decode before live := by
  constructor
  all_goals intro p success; rw [failed] at success; cases success

theorem complete (h : Call) (kind : Kind) (api : API h kind β γ) (laws : CopyLaws api)
    (tensors : RegionName → Host.Target) (encode : Bool → γ → Fin 4 → β)
    (grid : Prepared h kind γ → Heap γ → Option (Heap γ))
    (crop : Prepared h kind γ → Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (reference : RegionName → Fin (Logical.cells h) → γ) (before : Copyback.Heap β)
    (works : ∀ p, api.prepare=.ok p → ∃ final, grid p p.heap=some final ∧
      ∀ r∈Roles.outputs kind, ∀ i, crop p final r i=reference r i) :
    Host.Outcome h kind tensors encode reference before (run h kind api tensors encode grid crop before) := by
  unfold run
  by_cases empty : Logical.cells h=0
  · rw [if_pos empty]
    refine ⟨rfl,rfl,?_,fun _ _ => rfl⟩
    intro addr
    have noJobs (r : RegionName) : Host.roleJobs h tensors encode reference r=[] := by
      unfold Host.roleJobs Copyback.jobs
      have noIndices : List.finRange (Logical.cells h)=[] := by simp [empty]
      rw [noIndices]
      rfl
    have allEmpty : Host.allJobs h kind tensors encode reference=[] :=
      List.flatMap_eq_nil_iff.mpr (fun r _ => noJobs r)
    rw [allEmpty]
    rfl
  · rw [if_neg empty]
    cases prepared : api.prepare with
    | error message => exact ⟨Or.inl rfl,fun _ _ => rfl⟩
    | ok p =>
      apply Host.complete (preparedAPI api p) (copy_laws api laws p)
      intro heap heapEq
      have same : heap=p.heap := (Except.ok.inj heapEq).symm
      subst heap
      exact works p prepared

#axiomsClean preparation_failure
#axiomsClean preparation_laws_on_failure
#axiomsClean complete
end
end VeriTile.Bench.Repairs.DecayCumsum.WholeHost
