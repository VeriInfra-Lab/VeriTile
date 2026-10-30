import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LogicalStorage
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Copyback

/-! Caller-observable wrapper semantics. Preparation allocates independent
padded snapshots of the original read roles. Setup may fail at either launch;
all packed work precedes caller writes. A failure there leaves caller-owned
storage unchanged. Successful setup never supplies a source execution result:
the Real/FP entries instantiate the proved complete grids.

The generic copy helper checks destination element counts; the public checked
shapes make these checks succeed. On the admitted contiguous destinations,
ordinary copy_ has the specified byte result, with source-order copies across
roles. A failed copy may partially modify its own destination; its complete
outside frame is retained, and earlier completed roles are not rolled back.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.Host
open VeriTile.Triton Logical PackedStorage
noncomputable section
set_option maxHeartbeats 5000000
variable {β γ : Type}

def order : Kind → List RegionName
  | .forward => ["go"]
  | .prepare => ["kg","qg"]
  | .backward => ["dqo","dko","dg"]

theorem order_member (kind : Kind) (r : RegionName) : r∈order kind ↔ r∈Roles.outputs kind := by
  cases kind <;> simp [order,Roles.outputs] <;> tauto

structure Target where
  object : Nat
  shape : List Nat
  address : Nat → Nat
  negative : Bool

def view (h : Call) (target : Target) : Copyback.View (Logical.cells h) :=
  ⟨target.object,fun i => target.address i.val,target.negative⟩

def roleJobs (h : Call) (targets : RegionName → Target) (encode : Bool → γ → Fin 4 → β)
    (values : RegionName → Fin (Logical.cells h) → γ) (r : RegionName) : List (Copyback.Job β) :=
  Copyback.jobs (view h (targets r)) encode (values r)

def allJobs (h : Call) (kind : Kind) (targets : RegionName → Target) (encode : Bool → γ → Fin 4 → β)
    (values : RegionName → Fin (Logical.cells h) → γ) : List (Copyback.Job β) :=
  (order kind).flatMap (roleJobs h targets encode values)

def touches (h : Call) (kind : Kind) (targets : RegionName → Target) (addr : Nat) : Prop :=
  ∃ r∈order kind, Copyback.touches (view h (targets r)) addr

inductive Stage where
  | validation | preparation | setup | source | reshape | copy
  deriving DecidableEq
structure Failure (β : Type) where
  stage : Stage
  message : String
  heap : Copyback.Heap β
structure Returned (β : Type) where
  result : Unit
  /-- The exact same supplied objects; no replacement tensor is returned. -/
  targets : RegionName → Target
  heap : Copyback.Heap β

structure API (β γ : Type) where
  prepare : Except String (PackedStorage.Heap γ)
  setup : Except String Unit
  copy : RegionName → List (Copyback.Job β) → Copyback.Heap β → Except (Failure β) (Copyback.Heap β)

/-- Ordinary tensor copy laws, not numerical or source-execution assumptions.
The failure frame permits partial copy writes, including overlapping views. -/
structure CopyLaws (api : API β γ) : Prop where
  success : ∀ r jobs before after, api.copy r jobs before=.ok after → after=Copyback.commit jobs before
  failure : ∀ r jobs before error, api.copy r jobs before=.error error → error.stage=.copy ∧
    ∀ addr, (∀ job∈jobs, ¬(job.target≤addr ∧ addr<job.target+4)) → error.heap addr=before addr

def copies (api : API β γ) (h : Call) (targets : RegionName → Target) (encode : Bool → γ → Fin 4 → β)
    (values : RegionName → Fin (Logical.cells h) → γ) : List RegionName → Copyback.Heap β → Except (Failure β) (Copyback.Heap β)
  | [],before => .ok before
  | r::rest,before => if (targets r).shape.prod=Logical.cells h then
      match api.copy r (roleJobs h targets encode values r) before with
      | .error error => .error error
      | .ok after => copies api h targets encode values rest after
    else .error ⟨.reshape,"destination element count differs from the logical result",before⟩

/-- All setups and packed executions occur before the first caller write.
The source argument is instantiated with a proved complete finite grid. -/
def run (api : API β γ) (h : Call) (kind : Kind) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (grid : PackedStorage.Heap γ → Option (PackedStorage.Heap γ))
    (crop : PackedStorage.Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β) : Except (Failure β) (Returned β) :=
  if Logical.cells h=0 then .ok ⟨(),targets,before⟩ else
    match api.prepare with
    | .error message => .error ⟨.preparation,message,before⟩
    | .ok packed => match api.setup with
      | .error message => .error ⟨.setup,message,before⟩
      | .ok () => match grid packed with
        | none => .error ⟨.source,"source execution failed",before⟩
        | some final => match copies api h targets encode (crop final) (order kind) before with
          | .error error => .error error
          | .ok after => .ok ⟨(),targets,after⟩

theorem empty_return (api : API β γ) (h : Call) (kind : Kind) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (grid : PackedStorage.Heap γ → Option (PackedStorage.Heap γ))
    (crop : PackedStorage.Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β) (empty : Logical.cells h=0) :
    run api h kind targets encode grid crop before=.ok ⟨(),targets,before⟩ := by simp [run,empty]

theorem copies_success (api : API β γ) (laws : CopyLaws api) (h : Call) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (values : RegionName → Fin (Logical.cells h) → γ)
    (roles : List RegionName) (before after : Copyback.Heap β)
    (success : copies api h targets encode values roles before=.ok after) :
    after=Copyback.commit (roles.flatMap (roleJobs h targets encode values)) before := by
  induction roles generalizing before with
  | nil => cases success; rfl
  | cons r rest ih =>
    unfold copies at success
    split at success
    · cases copyEq : api.copy r (roleJobs h targets encode values r) before with
      | error e => simp [copyEq] at success
      | ok mid =>
        simp only [copyEq] at success
        rw [ih mid success,laws.success r _ before mid copyEq]
        simp [Copyback.commit,List.foldl_append]
    · cases success

theorem copies_frame (api : API β γ) (laws : CopyLaws api) (h : Call) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (values : RegionName → Fin (Logical.cells h) → γ)
    (roles : List RegionName) (before : Copyback.Heap β) (addr : Nat)
    (outside : ∀ r∈roles, ¬Copyback.touches (view h (targets r)) addr) :
    match copies api h targets encode values roles before with
    | .ok after => after addr=before addr
    | .error error => (error.stage=.reshape ∨ error.stage=.copy) ∧ error.heap addr=before addr := by
  induction roles generalizing before with
  | nil => rfl
  | cons r rest ih =>
    have restOutside := fun s hs => outside s (List.mem_cons_of_mem r hs)
    have copyOutside : ∀ job∈roleJobs h targets encode values r, ¬(job.target≤addr ∧ addr<job.target+4) := by
      intro job member bounds
      obtain ⟨i,_,rfl⟩ := List.mem_map.mp member
      exact outside r (by simp) ⟨i,bounds⟩
    by_cases shape : (targets r).shape.prod=Logical.cells h
    · simp only [copies,if_pos shape]
      cases copyEq : api.copy r (roleJobs h targets encode values r) before with
      | error e =>
        obtain ⟨stage,frame⟩ := laws.failure r _ before e copyEq
        exact ⟨Or.inr stage,frame addr copyOutside⟩
      | ok mid =>
        have midEq : mid addr=before addr := by
          rw [laws.success r _ before mid copyEq]
          exact Copyback.frame _ _ _ copyOutside
        have recursive := ih mid restOutside
        simp only []
        cases next : copies api h targets encode values rest mid with
        | ok final => rw [next] at recursive; exact recursive.trans midEq
        | error e => rw [next] at recursive; exact ⟨recursive.1,recursive.2.trans midEq⟩
    · simp only [copies,if_neg shape]
      simp

theorem jobs_congr (h : Call) (kind : Kind) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (a b : RegionName → Fin (Logical.cells h) → γ)
    (same : ∀ r∈Roles.outputs kind, ∀ i, a r i=b r i) :
    allJobs h kind targets encode a=allJobs h kind targets encode b := by
  unfold allJobs
  apply List.flatMap_congr
  intro r hr
  have equal : a r=b r := funext (same r ((order_member kind r).mp hr))
  simp only [roleJobs,equal]

theorem successful_result (api : API β γ) (laws : CopyLaws api) (h : Call) (kind : Kind)
    (targets : RegionName → Target) (encode : Bool → γ → Fin 4 → β)
    (grid : PackedStorage.Heap γ → Option (PackedStorage.Heap γ))
    (crop : PackedStorage.Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (reference : RegionName → Fin (Logical.cells h) → γ) (before : Copyback.Heap β)
    (works : ∀ packed, api.prepare=.ok packed → ∃ final, grid packed=some final ∧
      ∀ r∈Roles.outputs kind, ∀ i, crop final r i=reference r i)
    (after : Returned β) (success : run api h kind targets encode grid crop before=.ok after) :
    after.result=() ∧ after.targets=targets ∧
      ∀ addr, after.heap addr=(Copyback.lastValue (allJobs h kind targets encode reference) addr).getD (before addr) := by
  by_cases empty : Logical.cells h=0
  · rw [empty_return api h kind targets encode grid crop before empty] at success
    cases success
    refine ⟨rfl,rfl,?_⟩
    intro addr
    have noJobs (r : RegionName) : roleJobs h targets encode reference r=[] := by
      unfold roleJobs Copyback.jobs
      have noIndices : List.finRange (Logical.cells h)=[] := by simp [empty]
      rw [noIndices]
      rfl
    have allEmpty : allJobs h kind targets encode reference=[] :=
      List.flatMap_eq_nil_iff.mpr (fun r _ => noJobs r)
    rw [allEmpty]
    rfl
  · unfold run at success
    rw [if_neg empty] at success
    cases prep : api.prepare with
    | error e => simp [prep] at success
    | ok packed =>
      obtain ⟨final,he,out⟩ := works packed prep
      cases setup : api.setup with
      | error e => simp [prep,setup] at success
      | ok unit =>
        cases unit
        simp only [prep,setup,he] at success
        cases copyEq : copies api h targets encode (crop final) (order kind) before with
        | error e => simp [copyEq] at success
        | ok result =>
          simp only [copyEq,Except.ok.injEq] at success
          subst after
          refine ⟨rfl,rfl,?_⟩
          intro addr
          have eq := copies_success api laws h targets encode (crop final) (order kind) before result copyEq
          change result addr=_
          rw [eq,Copyback.exact_last_write]
          exact congrArg (fun js => (Copyback.lastValue js addr).getD (before addr)) (jobs_congr h kind targets encode (crop final) reference out)

#axiomsClean empty_return
#axiomsClean copies_frame
#axiomsClean successful_result
end
end VeriTile.Bench.Repairs.DecayCumsum.Host
