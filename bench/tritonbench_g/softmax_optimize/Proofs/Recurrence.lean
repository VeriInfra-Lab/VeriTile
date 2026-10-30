import bench.tritonbench_g.softmax_optimize.Proofs.Config
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_LegacyReal

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Recurrence
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton
noncomputable section
local instance : DecidableEq (WithBot ℝ) := Classical.decEq _

/-- Only source-local operations and one exact -infinity test. There is no
execution, output, memory, index or softmax equality field. -/
structure Operations (α : Type) where
  zero : α
  negInf : α
  maximum : α → α → α
  subtract : α → α → α
  exponential : α → α
  multiply : α → α → α
  add : α → α → α
  isNegInf : α → Bool

/-- The candidate's exact three guarded mass statements. -/
def step {α : Type} (ops : Operations α) (state : α × α) (input : α) : α × α :=
  let newMax := ops.maximum state.1 input
  let oldMass := if ops.isNegInf newMax then state.2
    else ops.multiply (ops.exponential (ops.subtract state.1 newMax)) state.2
  let newMass := if ops.isNegInf input then ops.zero
    else ops.exponential (ops.subtract input newMax)
  (newMax, ops.add oldMass newMass)

/-- Row-independent source lane input, including the explicit masked -inf. -/
def laneInput {α : Type} (ops : Operations α) (input : Nat → α)
    (N B tileIndex lane : Nat) : α :=
  if tileIndex * B + lane < N then input (tileIndex * B + lane) else ops.negInf

def laneAfter {α : Type} (ops : Operations α) (input : Nat → α)
    (N B lane count : Nat) : α × α :=
  (List.range count).foldl (fun state tileIndex =>
    step ops state (laneInput ops input N B tileIndex lane)) (ops.negInf, ops.zero)

theorem laneAfter_succ {α : Type} (ops : Operations α) (input : Nat → α)
    (N B lane count : Nat) :
    laneAfter ops input N B lane (count + 1) =
      step ops (laneAfter ops input N B lane count) (laneInput ops input N B count lane) := by
  simp [laneAfter, List.range_succ, List.foldl_append]

/-- Source pass A followed by the unique tail pass B. -/
def twoLoops {α : Type} (ops : Operations α) (input : Nat → α)
    (N B lane : Nat) : α × α :=
  let priorState := laneAfter ops input N B lane (Config.fullTiles N B)
  step ops priorState (laneInput ops input N B (Config.fullTiles N B) lane)

theorem two_loops_complete {α : Type} (ops : Operations α) (input : Nat → α)
    (N B lane : Nat) :
    twoLoops ops input N B lane = laneAfter ops input N B lane (Config.fullTiles N B + 1) := by
  simp [twoLoops, laneAfter, List.range_succ, List.foldl_append]

def realOperations : Operations (WithBot ℝ) where
  zero := (0 : ℝ)
  negInf := ⊥
  maximum := (· ⊔ ·)
  subtract := WithBot.realSub
  exponential := WithBot.realExp
  multiply := WithBot.realMul
  add := WithBot.realAdd
  isNegInf := fun value => decide (value = ⊥)

def embed (state : WithBot ℝ × ℝ) : WithBot ℝ × WithBot ℝ :=
  (state.1, (state.2 : WithBot ℝ))

/-- On a genuine real input, the candidate reuses the original proven online
recurrence. The equality is proved, not installed as a numerical port. -/
theorem step_real_input (m : WithBot ℝ) (z x : ℝ) :
    step realOperations (embed (m, z)) (x : WithBot ℝ) = embed (soStep (m, z) x) := by
  have hm : m ⊔ (x : WithBot ℝ) ≠ ⊥ := by cases m <;> simp
  apply Prod.ext
  · rfl
  · simp only [step, realOperations, embed, decide_eq_true_eq, hm,
      WithBot.coe_ne_bot, ↓reduceIte]
    exact soStep_snd_match m z x

/-- Padding is an identity update, including the never-active initial state.
The candidate guards retain existing NaN mass in the separate FP model. -/
theorem step_real_padding (m : WithBot ℝ) (z : ℝ) :
    step realOperations (embed (m, z)) ⊥ = embed (m, z) := by
  cases m with
  | bot =>
    simp only [step, realOperations, embed, sup_bot_eq, decide_eq_true_eq, ↓reduceIte]
    change ((⊥ : WithBot ℝ), WithBot.realAdd (z : WithBot ℝ) (0 : ℝ)) = (⊥, (z : WithBot ℝ))
    rw [WithBot.realAdd_coe_coe, add_zero]
  | coe r =>
    simp only [step, realOperations, embed, sup_bot_eq, decide_eq_true_eq,
      WithBot.coe_ne_bot, ↓reduceIte]
    change ((r : WithBot ℝ), WithBot.realAdd
      (WithBot.realMul (WithBot.realExp (WithBot.realSub (r : WithBot ℝ) (r : WithBot ℝ)))
        (z : WithBot ℝ)) (0 : ℝ)) = ((r : WithBot ℝ), (z : WithBot ℝ))
    rw [WithBot.realSub_coe_coe, sub_self, WithBot.realExp_coe,
      Real.exp_zero, WithBot.realMul_coe_coe, one_mul,
      WithBot.realAdd_coe_coe, add_zero]

theorem fold_real_inputs (xs : List (Option ℝ)) (m : WithBot ℝ) (z : ℝ) :
    xs.foldl (step realOperations) (embed (m, z)) =
      embed ((xs.filterMap id).foldl soStep (m, z)) := by
  induction xs generalizing m z with
  | nil => rfl
  | cons head tail ih =>
    cases head with
    | none =>
      change tail.foldl (step realOperations) (step realOperations (embed (m, z)) ⊥) = _
      rw [step_real_padding]
      exact ih m z
    | some x =>
      change tail.foldl (step realOperations)
        (step realOperations (embed (m, z)) (x : WithBot ℝ)) =
        embed ((tail.filterMap id).foldl soStep (soStep (m, z) x))
      rw [step_real_input]
      exact ih (soStep (m, z) x).1 (soStep (m, z) x).2

theorem nonempty_operations : Nonempty (Operations (WithBot ℝ)) := ⟨realOperations⟩

/-- The guarded lane fold absorbs precisely the original logical scores. -/
theorem laneAfter_real (s : BlockState) (X : RegionName) (N B count : Nat)
    (lane : Fin B) :
    laneAfter realOperations (fun j => (s.readMem X (s.pid * N + j) : WithBot ℝ))
      N B lane count = embed (laneState s X N B lane count) := by
  have hi : ∀ i, laneInput realOperations
      (fun j => (s.readMem X (s.pid * N + j) : WithBot ℝ)) N B i lane =
      laneScoreAt s X N B lane i := by
    intro i
    unfold laneInput laneScoreAt
    split <;> rfl
  have h := fold_real_inputs ((List.range count).map (laneScoreAt s X N B lane)) ⊥ 0
  change ((List.range count).map (fun i => (laneScoreAt s X N B lane i : WithBot ℝ))).foldl
    (step realOperations) (embed (⊥, 0)) = _ at h
  erw [List.foldl_map] at h
  unfold laneAfter
  simp_rw [hi]
  simpa only [List.filterMap_map, Function.comp_def, id_eq,
    laneState, laneScores, embed, realOperations] using h

theorem twoLoops_real (s : BlockState) (X : RegionName) (N B : Nat) (lane : Fin B) :
    twoLoops realOperations (fun j => (s.readMem X (s.pid * N + j) : WithBot ℝ))
      N B lane = embed (laneState s X N B lane (Config.fullTiles N B + 1)) := by
  rw [two_loops_complete, laneAfter_real]

/-- The guarded recurrence has the same proved closed denominator, without
assuming that any source execution or output has already succeeded. -/
theorem twoLoops_mass (s : BlockState) (X : RegionName) (N B : Nat) (lane : Fin B) :
    (twoLoops realOperations (fun j => (s.readMem X (s.pid * N + j) : WithBot ℝ))
      N B lane).2 =
      (((laneState s X N B lane (Config.fullTiles N B + 1)).1.elim 0
        (fun r => Real.exp (-r))) *
        ((laneScores s X N B lane (Config.fullTiles N B + 1)).map Real.exp).sum : ℝ) := by
  rw [twoLoops_real]
  change ((laneState s X N B lane _).2 : WithBot ℝ) = _
  rw [laneState_denom_closed]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Recurrence
