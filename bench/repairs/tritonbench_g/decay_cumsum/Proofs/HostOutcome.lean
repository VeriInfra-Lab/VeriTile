import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Host

namespace VeriTile.Bench.Repairs.DecayCumsum.Host
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
variable {β γ : Type}

theorem jobs_frame (h : Call) (kind : Kind) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (values : RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β) (addr : Nat) (outside : ¬touches h kind targets addr) :
    Copyback.commit (allJobs h kind targets encode values) before addr=before addr := by
  apply Copyback.frame
  intro job member bounds
  obtain ⟨r,hr,hj⟩ := List.mem_flatMap.mp member
  obtain ⟨i,_,rfl⟩ := List.mem_map.mp hj
  exact outside ⟨r,hr,i,bounds⟩

theorem copies_failure_stage (api : API β γ) (laws : CopyLaws api) (h : Call) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (values : RegionName → Fin (Logical.cells h) → γ)
    (roles : List RegionName) (before : Copyback.Heap β) (error : Failure β)
    (failure : copies api h targets encode values roles before=.error error) :
    error.stage=.reshape ∨ error.stage=.copy := by
  induction roles generalizing before with
  | nil => cases failure
  | cons r rest ih =>
    unfold copies at failure
    split at failure
    · cases copyEq : api.copy r (roleJobs h targets encode values r) before with
      | error e =>
        have same : e=error := by simpa only [copyEq,Except.error.injEq] using failure
        subst e
        exact Or.inr (laws.failure r _ before error copyEq).1
      | ok mid =>
        simp only [copyEq] at failure
        exact ih mid failure
    · cases failure
      exact Or.inl rfl

def OrdinaryFailure (stage : Stage) : Prop :=
  stage=.preparation ∨ stage=.setup ∨ stage=.reshape ∨ stage=.copy

theorem failure_result (api : API β γ) (laws : CopyLaws api) (h : Call) (kind : Kind)
    (targets : RegionName → Target) (encode : Bool → γ → Fin 4 → β)
    (grid : PackedStorage.Heap γ → Option (PackedStorage.Heap γ))
    (crop : PackedStorage.Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β)
    (works : ∀ packed, api.prepare=.ok packed → ∃ final, grid packed=some final)
    (error : Failure β) (failure : run api h kind targets encode grid crop before=.error error) :
    OrdinaryFailure error.stage ∧ ∀ addr, ¬touches h kind targets addr → error.heap addr=before addr := by
  by_cases empty : Logical.cells h=0
  · simp [run,empty] at failure
  · unfold run at failure
    rw [if_neg empty] at failure
    cases prep : api.prepare with
    | error message =>
      simp only [prep,Except.error.injEq] at failure
      subst error
      exact ⟨Or.inl rfl,fun _ _ => rfl⟩
    | ok packed =>
      obtain ⟨final,he⟩ := works packed prep
      cases setup : api.setup with
      | error message =>
        simp only [prep,setup,Except.error.injEq] at failure
        subst error
        exact ⟨Or.inr (Or.inl rfl),fun _ _ => rfl⟩
      | ok done =>
        cases done
        simp only [prep,setup,he] at failure
        cases copyEq : copies api h targets encode (crop final) (order kind) before with
        | ok result => simp [copyEq] at failure
        | error e =>
          have same : e=error := by simpa only [copyEq,Except.error.injEq] using failure
          subst e
          refine ⟨Or.inr (Or.inr (copies_failure_stage api laws h targets encode (crop final) (order kind) before error copyEq)),?_⟩
          intro addr outside
          have frame := copies_frame api laws h targets encode (crop final) (order kind) before addr
            (fun r hr inside => outside ⟨r,hr,inside⟩)
          rw [copyEq] at frame
          exact frame.2

def Outcome (h : Call) (kind : Kind) (targets : RegionName → Target)
    (encode : Bool → γ → Fin 4 → β) (reference : RegionName → Fin (Logical.cells h) → γ)
    (before : Copyback.Heap β) : Except (Failure β) (Returned β) → Prop
  | .ok after => after.result=() ∧ after.targets=targets ∧
      (∀ addr, after.heap addr=(Copyback.lastValue (allJobs h kind targets encode reference) addr).getD (before addr)) ∧
      (∀ addr, ¬touches h kind targets addr → after.heap addr=before addr)
  | .error error => OrdinaryFailure error.stage ∧
      ∀ addr, ¬touches h kind targets addr → error.heap addr=before addr

theorem complete (api : API β γ) (laws : CopyLaws api) (h : Call) (kind : Kind)
    (targets : RegionName → Target) (encode : Bool → γ → Fin 4 → β)
    (grid : PackedStorage.Heap γ → Option (PackedStorage.Heap γ))
    (crop : PackedStorage.Heap γ → RegionName → Fin (Logical.cells h) → γ)
    (reference : RegionName → Fin (Logical.cells h) → γ) (before : Copyback.Heap β)
    (works : ∀ packed, api.prepare=.ok packed → ∃ final, grid packed=some final ∧
      ∀ r∈Roles.outputs kind, ∀ i, crop final r i=reference r i) :
    Outcome h kind targets encode reference before (run api h kind targets encode grid crop before) := by
  cases he : run api h kind targets encode grid crop before with
  | error error =>
    exact failure_result api laws h kind targets encode grid crop before
      (fun packed hp => by obtain ⟨final,run,_⟩ := works packed hp; exact ⟨final,run⟩) error he
  | ok after =>
    obtain ⟨none,identities,bytes⟩ := successful_result api laws h kind targets encode grid crop reference before works after he
    refine ⟨none,identities,bytes,?_⟩
    intro addr outside
    rw [bytes,←Copyback.exact_last_write]
    exact jobs_frame h kind targets encode reference before addr outside

#axiomsClean complete
end
end VeriTile.Bench.Repairs.DecayCumsum.Host
