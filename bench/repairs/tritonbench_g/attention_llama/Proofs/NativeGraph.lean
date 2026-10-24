import bench.repairs.tritonbench_g.attention_llama.Proofs.NativeSchedule

/-! Recorded CUDA SM80 graph, using only scalar and fixed K16 ISA ports.
The source-to-recorded-graph compiler boundary is external and audited.
Native numerical transport does not assume a whole reduction or dot law. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.NativeGraph
open NativeSchedule

structure Ops (α : Type) where
  zero : α
  one : α
  negativeInfinity : α
  log2e : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  maximum : α → α → α
  ex2Approx : α → α
  half : α → α
  e5m2ToHalf : UInt8 → α
  fma : α → α → α → α
  mma16 : (Fin 16 → α) → (Fin 16 → α) → α → α

/-- Distinct physical lane results are retained; no commutativity, NaN, or
signed-zero law identifies the four shuffle results. -/
structure State (α : Type) (D : Nat) where
  maximum : Fin 4 → α
  denominator : Fin 4 → α
  accumulator : Fin D → α

def State.map {α β : Type} {D : Nat} (f : α → β) (s : State α D) : State β D :=
  ⟨fun l => f (s.maximum l), fun l => f (s.denominator l), fun d => f (s.accumulator d)⟩

def initial {α : Type} (A : Ops α) (D : Nat) : State α D :=
  ⟨fun _ => A.negativeInfinity, fun _ => A.zero, fun _ => A.zero⟩

def key {α : Type} {D : Nat} (A : Ops α) (input : SourceGraph.Inputs α D)
    (fp8 : Bool) (N k : Nat) (d : Fin D) : α :=
  if fp8 then A.e5m2ToHalf (if k < N then input.byteKey k d else 0)
  else if k < N then input.halfKey k d else A.zero

def value {α : Type} {D : Nat} (A : Ops α) (input : SourceGraph.Inputs α D)
    (fp8 : Bool) (N k : Nat) (d : Fin D) : α :=
  if fp8 then A.e5m2ToHalf (if k < N then input.byteValue k d else 0)
  else if k < N then input.halfValue k d else A.zero

def padded {α : Type} {D : Nat} (zero : α) (x : Fin D → α) (n : Nat) : α :=
  if h : n < D then x ⟨n,h⟩ else zero

def exp {α : Type} (A : Ops α) (x : α) : α := A.ex2Approx (A.mul x A.log2e)

def score {α : Type} {D : Nat} (A : Ops α) (input : SourceGraph.Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position query block : Nat) (col : Fin 64) : α :=
  let k := block * 64 + col.val
  let qk := mmaFold A.mma16 (mmaCoordinate fp8 D) (padded A.zero input.query)
    (padded A.zero (key A input fp8 N k)) A.zero (D/16)
  let scaled := A.mul scale qk
  let masked := if k < N then scaled else A.negativeInfinity
  if causal then if k + position ≤ query then masked else A.negativeInfinity else masked

def step {α : Type} {D : Nat} (A : Ops α) (input : SourceGraph.Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position query block : Nat)
    (state : State α D) : State α D :=
  let qk := score A input scale fp8 causal N position query block
  let m := fun l => A.maximum ((selected D l).eval A.maximum qk) (state.maximum l)
  let alpha := fun l => exp A (A.sub (state.maximum l) (m l))
  let previous := fun l => A.mul (state.denominator l) (alpha l)
  let probability := fun col : Fin 64 => exp A (A.sub (qk col) (m (lane col.val)))
  let current := fun l => A.fma (state.denominator l) (alpha l) ((selected D l).eval A.add probability)
  let reciprocal := fun l => A.div A.one (current l)
  let halfProbability := fun col : Fin 64 => A.half (A.mul (probability col) (reciprocal (lane col.val)))
  { maximum := m
    denominator := current
    accumulator := fun d => mmaFold A.mma16 (mmaCoordinate fp8 64)
      (fun k => halfProbability (column k))
      (fun k => value A input fp8 N (block * 64 + (column k).val) d)
      (A.mul (state.accumulator d) (A.mul (previous (lane d.val)) (reciprocal (lane d.val)))) 4 }

def run {α : Type} {D : Nat} (A : Ops α) (input : SourceGraph.Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position query : Nat) : Nat → State α D
  | 0 => initial A D
  | n + 1 => step A input scale fp8 causal N position query n
      (run A input scale fp8 causal N position query n)

def returnedWord {α : Type} {D : Nat} (A : Ops α) (input : SourceGraph.Inputs α D)
    (scale : α) (fp8 causal : Bool) (N position pid row : Nat) (d : Fin D) : α :=
  A.half ((run A input scale fp8 causal N position (pid * 64 + row)
    (MaskCoverage.count (MaskCoverage.stop N causal position pid))).accumulator d)

structure Agreement {α β : Type} (A : Ops α) (B : Ops β) (f : α → β) : Prop where
  zero : f A.zero = B.zero
  one : f A.one = B.one
  negativeInfinity : f A.negativeInfinity = B.negativeInfinity
  log2e : f A.log2e = B.log2e
  add : ∀ x y, f (A.add x y) = B.add (f x) (f y)
  sub : ∀ x y, f (A.sub x y) = B.sub (f x) (f y)
  mul : ∀ x y, f (A.mul x y) = B.mul (f x) (f y)
  div : ∀ x y, f (A.div x y) = B.div (f x) (f y)
  maximum : ∀ x y, f (A.maximum x y) = B.maximum (f x) (f y)
  ex2Approx : ∀ x, f (A.ex2Approx x) = B.ex2Approx (f x)
  half : ∀ x, f (A.half x) = B.half (f x)
  e5m2ToHalf : ∀ x, f (A.e5m2ToHalf x) = B.e5m2ToHalf x
  fma : ∀ x y z, f (A.fma x y z) = B.fma (f x) (f y) (f z)
  mma16 : ∀ (x y : Fin 16 → α) z,
    f (A.mma16 x y z) = B.mma16 (fun i => f (x i)) (fun i => f (y i)) (f z)

theorem key_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : SourceGraph.Inputs α D) (fp8 : Bool) (N k : Nat) (d : Fin D) :
    f (key A input fp8 N k d) = key B (input.map f) fp8 N k d := by
  cases fp8 <;> by_cases hk : k < N <;>
    simp [key, SourceGraph.Inputs.map, hk, h.zero, h.e5m2ToHalf]

theorem value_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : SourceGraph.Inputs α D) (fp8 : Bool) (N k : Nat) (d : Fin D) :
    f (value A input fp8 N k d) = value B (input.map f) fp8 N k d := by
  cases fp8 <;> by_cases hk : k < N <;>
    simp [value, SourceGraph.Inputs.map, hk, h.zero, h.e5m2ToHalf]

theorem padded_map {α β : Type} {D : Nat} (f : α → β) (zero : α) (x : Fin D → α) (n : Nat) :
    f (padded zero x n) = padded (f zero) (fun i => f (x i)) n := by
  simp only [padded]; split <;> rfl

theorem score_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : SourceGraph.Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position query block : Nat) (col : Fin 64) :
    f (score A input scale fp8 causal N position query block col) =
      score B (input.map f) (f scale) fp8 causal N position query block col := by
  cases causal <;> by_cases hk : block * 64 + col.val < N <;>
    by_cases hc : block * 64 + col.val + position ≤ query <;>
    simp only [score, Bool.false_eq_true, ↓reduceIte, hk, hc,
      h.negativeInfinity, h.mul, mmaFold_map A.mma16 B.mma16 f h.mma16,
      padded_map, h.zero, key_map A B f h]
  all_goals rfl

theorem step_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : SourceGraph.Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position query block : Nat) (state : State α D) :
    (step A input scale fp8 causal N position query block state).map f =
      step B (input.map f) (f scale) fp8 causal N position query block (state.map f) := by
  simp only [step, State.map, exp, h.one, h.log2e, h.sub, h.mul, h.div,
    h.maximum, h.ex2Approx, h.half, h.fma, mmaFold_map A.mma16 B.mma16 f h.mma16,
    Tree.eval_map A.maximum B.maximum f h.maximum, Tree.eval_map A.add B.add f h.add,
    score_map A B f h, value_map A B f h]

theorem run_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : SourceGraph.Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position query n : Nat) :
    (run A input scale fp8 causal N position query n).map f =
      run B (input.map f) (f scale) fp8 causal N position query n := by
  induction n with
  | zero => simp [run, initial, State.map, h.zero, h.negativeInfinity]
  | succ n ih => rw [run, step_map A B f h, ih]; rfl

theorem returnedWord_map {α β : Type} {D : Nat} (A : Ops α) (B : Ops β) (f : α → β)
    (h : Agreement A B f) (input : SourceGraph.Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position pid row : Nat) (d : Fin D) :
    f (returnedWord A input scale fp8 causal N position pid row d) =
      returnedWord B (input.map f) (f scale) fp8 causal N position pid row d := by
  have hr := run_map A B f h input scale fp8 causal N position (pid * 64 + row)
    (MaskCoverage.count (MaskCoverage.stop N causal position pid))
  have hv := congrArg (fun s : State β D => s.accumulator d) hr
  simpa only [returnedWord, h.half, State.map] using congrArg B.half hv

end VeriTile.Bench.Repairs.AttentionLlama.NativeGraph
