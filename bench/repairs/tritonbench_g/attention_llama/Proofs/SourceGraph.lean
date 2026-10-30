import bench.repairs.tritonbench_g.attention_llama.Proofs.Geometry

/-! Explicit source instruction graph. Each arithmetic field is one scalar,
reduction, cast, or dot instruction; none returns an attention result. The
two source dot calls remain separate from their following additions. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.SourceGraph

structure Ops (α : Type) where
  zero : α
  one : α
  negativeInfinity : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  maximum : α → α → α
  exp : α → α
  half : α → α
  e5m2ToHalf : UInt8 → α
  dot : {n : Nat} → (Fin n → α) → (Fin n → α) → α
  sum64 : (Fin 64 → α) → α
  max64 : (Fin 64 → α) → α

structure Inputs (α : Type) (D : Nat) where
  query : Fin D → α
  halfKey : Nat → Fin D → α
  halfValue : Nat → Fin D → α
  byteKey : Nat → Fin D → UInt8
  byteValue : Nat → Fin D → UInt8

def Inputs.map {α β : Type} {D : Nat} (f : α → β) (x : Inputs α D) : Inputs β D :=
  { query := fun d => f (x.query d)
    halfKey := fun k d => f (x.halfKey k d)
    halfValue := fun k d => f (x.halfValue k d)
    byteKey := x.byteKey
    byteValue := x.byteValue }

structure State (α : Type) (D : Nat) where
  maximum : α
  denominator : α
  accumulator : Fin D → α

def State.map {α β : Type} {D : Nat} (f : α → β) (s : State α D) : State β D :=
  ⟨f s.maximum, f s.denominator, fun d => f (s.accumulator d)⟩

def initial {α : Type} (A : Ops α) (D : Nat) : State α D :=
  ⟨A.negativeInfinity, A.zero, fun _ => A.zero⟩

def key {α : Type} {D : Nat} (A : Ops α) (input : Inputs α D)
    (fp8 : Bool) (N k : Nat) (d : Fin D) : α :=
  if fp8 then A.e5m2ToHalf (if k < N then input.byteKey k d else 0)
  else if k < N then input.halfKey k d else A.zero

def value {α : Type} {D : Nat} (A : Ops α) (input : Inputs α D)
    (fp8 : Bool) (N k : Nat) (d : Fin D) : α :=
  if fp8 then A.e5m2ToHalf (if k < N then input.byteValue k d else 0)
  else if k < N then input.halfValue k d else A.zero

def score {α : Type} {D : Nat} (A : Ops α) (input : Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position query block : Nat) (lane : Fin 64) : α :=
  let k := block * 64 + lane.val
  let dot := A.dot input.query (key A input fp8 N k)
  let scaled := A.mul (A.add A.zero dot) scale
  let masked := if k < N then scaled else A.negativeInfinity
  if causal then if k + position ≤ query then masked else A.negativeInfinity else masked

def step {α : Type} {D : Nat} (A : Ops α) (input : Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position query block : Nat)
    (state : State α D) : State α D :=
  let qk := score A input scale fp8 causal N position query block
  let m := A.maximum (A.max64 qk) state.maximum
  let previous := A.mul state.denominator (A.exp (A.sub state.maximum m))
  let probability := fun lane => A.exp (A.sub (qk lane) m)
  let current := A.add (A.sum64 probability) previous
  let reciprocal := A.div A.one current
  let halfProbability := fun lane => A.half (A.mul (probability lane) reciprocal)
  { maximum := m
    denominator := current
    accumulator := fun d => A.add
      (A.mul (state.accumulator d) (A.mul previous reciprocal))
      (A.dot halfProbability (fun lane => value A input fp8 N (block * 64 + lane.val) d)) }

def run {α : Type} {D : Nat} (A : Ops α) (input : Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position query : Nat) : Nat → State α D
  | 0 => initial A D
  | n + 1 => step A input scale fp8 causal N position query n
      (run A input scale fp8 causal N position query n)

def returnedWord {α : Type} {D : Nat} (A : Ops α) (input : Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position pid lane : Nat) (d : Fin D) : α :=
  A.half ((run A input scale fp8 causal N position (pid * 64 + lane)
    (MaskCoverage.count (MaskCoverage.stop N causal position pid))).accumulator d)

/-- Exact local primitive correspondence. In a selected numerical instance
these fields are deferred individually and apply to its reached operands.
This generic transport theorem itself makes no hardware-conformance claim. -/
structure Agreement {α β : Type} (A : Ops α) (B : Ops β) (f : α → β) : Prop where
  zero : f A.zero = B.zero
  one : f A.one = B.one
  negativeInfinity : f A.negativeInfinity = B.negativeInfinity
  add : ∀ x y, f (A.add x y) = B.add (f x) (f y)
  sub : ∀ x y, f (A.sub x y) = B.sub (f x) (f y)
  mul : ∀ x y, f (A.mul x y) = B.mul (f x) (f y)
  div : ∀ x y, f (A.div x y) = B.div (f x) (f y)
  maximum : ∀ x y, f (A.maximum x y) = B.maximum (f x) (f y)
  exp : ∀ x, f (A.exp x) = B.exp (f x)
  half : ∀ x, f (A.half x) = B.half (f x)
  e5m2ToHalf : ∀ x, f (A.e5m2ToHalf x) = B.e5m2ToHalf x
  dot : ∀ {n} (x y : Fin n → α), f (A.dot x y) = B.dot (fun i => f (x i)) (fun i => f (y i))
  sum64 : ∀ x, f (A.sum64 x) = B.sum64 (fun i => f (x i))
  max64 : ∀ x, f (A.max64 x) = B.max64 (fun i => f (x i))

theorem key_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : Inputs α D) (fp8 : Bool) (N k : Nat) (d : Fin D) :
    f (key A input fp8 N k d) = key B (input.map f) fp8 N k d := by
  cases fp8 <;> by_cases hk : k < N <;>
    simp [key, Inputs.map, hk, h.zero, h.e5m2ToHalf]

theorem value_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : Inputs α D) (fp8 : Bool) (N k : Nat) (d : Fin D) :
    f (value A input fp8 N k d) = value B (input.map f) fp8 N k d := by
  cases fp8 <;> by_cases hk : k < N <;>
    simp [value, Inputs.map, hk, h.zero, h.e5m2ToHalf]

theorem score_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position query block : Nat) (lane : Fin 64) :
    f (score A input scale fp8 causal N position query block lane) =
      score B (input.map f) (f scale) fp8 causal N position query block lane := by
  cases causal <;> by_cases hk : block * 64 + lane.val < N <;>
    by_cases hc : block * 64 + lane.val + position ≤ query <;>
    simp only [score, Bool.false_eq_true, ↓reduceIte, hk, hc, h.negativeInfinity,
      h.mul, h.add, h.zero, h.dot, key_map A B f h]
  all_goals rfl

theorem step_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position query block : Nat) (state : State α D) :
    (step A input scale fp8 causal N position query block state).map f =
      step B (input.map f) (f scale) fp8 causal N position query block (state.map f) := by
  simp only [step, State.map,
    h.one, h.add, h.sub, h.mul, h.div,
    h.maximum, h.exp, h.half, h.dot, h.sum64, h.max64,
    score_map A B f h, value_map A B f h]

theorem run_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position query n : Nat) :
    (run A input scale fp8 causal N position query n).map f =
      run B (input.map f) (f scale) fp8 causal N position query n := by
  induction n with
  | zero => simp [run, initial, State.map, h.zero, h.negativeInfinity]
  | succ n ih => rw [run, step_map A B f h, ih]; rfl

theorem returnedWord_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position pid lane : Nat) (d : Fin D) :
    f (returnedWord A input scale fp8 causal N position pid lane d) =
      returnedWord B (input.map f) (f scale) fp8 causal N position pid lane d := by
  have hr := run_map A B f h input scale fp8 causal N position (pid * 64 + lane)
    (MaskCoverage.count (MaskCoverage.stop N causal position pid))
  have hv := congrArg (fun s : State β D => s.accumulator d) hr
  simpa only [returnedWord, h.half, State.map] using congrArg B.half hv

end VeriTile.Bench.Repairs.AttentionLlama.SourceGraph
