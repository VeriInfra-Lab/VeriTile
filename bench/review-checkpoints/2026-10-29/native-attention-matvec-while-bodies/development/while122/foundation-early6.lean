import Init

/-! Partial while-loop semantics with a finite-trace characterization.
The guard is evaluated in the current state on every iteration. This is a
least fixed point in Option, with no chosen fuel limit or termination premise.
Successful execution is equivalent to a finite sequence of successful guard
and body evaluations. -/
namespace VeriTile.Triton.Control.While

def run {σ : Type} (condition : σ → Option Bool) (body : σ → Option σ)
    (state : σ) : Option σ := do
  let active ← condition state
  if active then
    let next ← body state
    run condition body next
  else pure state
partial_fixpoint

inductive Finishes {σ : Type} (condition : σ → Option Bool) (body : σ → Option σ) : σ → σ → Prop where
  | stop {state} : condition state = some false → Finishes condition body state state
  | step {state next final} : condition state = some true → body state = some next →
      Finishes condition body next final → Finishes condition body state final

theorem finite_execution {σ : Type} {condition : σ → Option Bool} {body : σ → Option σ}
    {state final : σ} (trace : Finishes condition body state final) :
    run condition body state = some final := by
  induction trace with
  | stop conditionFalse => rw [run]; simp [conditionFalse]
  | step conditionTrue bodyRuns _ ih => rw [run]; simp [conditionTrue, bodyRuns, ih]

theorem successful_execution_is_finite {σ : Type} (condition : σ → Option Bool)
    (body : σ → Option σ) (state final : σ) :
    run condition body state = some final → Finishes condition body state final := by
  apply run.partial_correctness condition body (Finishes condition body)
  intro approximate ih state final result
  cases hc : condition state with
  | none => simp [hc] at result
  | some active =>
    cases active with
    | false =>
      simp [hc] at result
      cases result
      exact .stop hc
    | true =>
      cases hb : body state with
      | none => simp [hc, hb] at result
      | some next =>
        simp [hc, hb] at result
        exact .step hc hb (ih next final result)

theorem execution_iff_finite {σ : Type} (condition : σ → Option Bool)
    (body : σ → Option σ) (state final : σ) :
    run condition body state = some final ↔ Finishes condition body state final :=
  ⟨successful_execution_is_finite condition body state final, finite_execution⟩

theorem never_stops_has_no_result {σ : Type} (body : σ → Option σ) (state : σ) :
    run (fun _ => some true) body state = none := by
  have noTrace : ∀ first final, ¬ Finishes (fun _ => some true) body first final := by
    intro first final trace
    induction trace with
    | stop impossible => simp at impossible
    | step _ _ _ ih => exact ih
  cases h : run (fun _ => some true) body state with
  | none => rfl
  | some final => exact False.elim (noTrace state final
      (successful_execution_is_finite _ body state final h))

theorem trace_simulation {α β : Type}
    {conditionA : α → Option Bool} {bodyA : α → Option α}
    {conditionB : β → Option Bool} {bodyB : β → Option β}
    (related : α → β → Prop)
    (conditions : ∀ a b, related a b → conditionA a = conditionB b)
    (bodies : ∀ a b, related a b → ∀ next, bodyA a = some next →
      ∃ other, bodyB b = some other ∧ related next other)
    {first final : α} (trace : Finishes conditionA bodyA first final) :
    ∀ other, related first other →
      ∃ result, Finishes conditionB bodyB other result ∧ related final result := by
  induction trace with
  | stop hc =>
    intro other hr
    exact ⟨other, .stop ((conditions _ _ hr).symm.trans hc), hr⟩
  | step hc hb _ ih =>
    intro other hr
    obtain ⟨next, hn, hrel⟩ := bodies _ _ hr _ hb
    obtain ⟨result, htrace, hresult⟩ := ih next hrel
    exact ⟨result, .step ((conditions _ _ hr).symm.trans hc) hn htrace, hresult⟩

theorem run_simulation {α β : Type}
    {conditionA : α → Option Bool} {bodyA : α → Option α}
    {conditionB : β → Option Bool} {bodyB : β → Option β}
    (related : α → β → Prop)
    (conditions : ∀ a b, related a b → conditionA a = conditionB b)
    (bodies : ∀ a b, related a b → ∀ next, bodyA a = some next →
      ∃ other, bodyB b = some other ∧ related next other)
    {first final : α} {other : β} (initial : related first other)
    (executed : run conditionA bodyA first = some final) :
    ∃ result, run conditionB bodyB other = some result ∧ related final result := by
  obtain ⟨result, trace, finalRelated⟩ := trace_simulation related conditions bodies
    (successful_execution_is_finite conditionA bodyA first final executed) other initial
  exact ⟨result, finite_execution trace, finalRelated⟩

end VeriTile.Triton.Control.While
