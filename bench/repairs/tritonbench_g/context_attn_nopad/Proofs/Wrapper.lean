import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Source

/-! Ordinary API outcomes in the candidate's actual order. In particular,
normalization precedes assertions; max and item occur only for nonempty metadata.
The result retains the state at the first failed operation. Local API contracts
describe coordinate preparation and caller frames, never kernel success/results.
This follows the repository's BmmChunkFwd.Wrapper pattern, adapted to this source.
-/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.Wrapper

inductive Stage where
  | resolveQ | resolveK | resolveV
  | resolveStart | contiguousStart | resolveLength | contiguousLength
  | resolveOut | maximum | item | toPythonInt | deviceSetup | copyback
  deriving DecidableEq

def normalization : List Stage :=
  [.resolveQ,.resolveK,.resolveV,.resolveStart,.contiguousStart,
    .resolveLength,.contiguousLength,.resolveOut]

def synchronization (batches : Nat) : List Stage :=
  if batches = 0 then [] else [.maximum,.item,.toPythonInt]

structure Request where
  qShape : List Nat
  kShape : List Nat
  vShape : List Nat
  lengthShape : List Nat
  maxInputLen : Nat

inductive ValidationError where
  | queryLastDimension | keyLastDimension | valueLastDimension
  | equalDimensions | supportedDimension | metadataBatchDimension | queryHeadDimension
  deriving DecidableEq

structure Dimensions where
  width : Nat
  batches : Nat
  heads : Nat
  deriving DecidableEq

def lookup (error : ValidationError) : Option Nat → Except ValidationError Nat
  | none => .error error
  | some value => .ok value

/-- Python evaluates the three trailing shapes, then both assertions, then
the batch/head shape pair. Binary64 scale calculation is total at the admitted
positive widths and has the separately pinned source values in Call.scale. -/
def validate (r : Request) : Except ValidationError Dimensions := do
  let q ← lookup .queryLastDimension r.qShape.getLast?
  let k ← lookup .keyLastDimension r.kShape.getLast?
  let v ← lookup .valueLastDimension r.vShape.getLast?
  if ¬ (q = k ∧ k = v) then throw .equalDimensions
  if ¬ (k = 16 ∨ k = 32 ∨ k = 64 ∨ k = 128) then throw .supportedDimension
  let batches ← lookup .metadataBatchDimension r.lengthShape[0]?
  let heads ← lookup .queryHeadDimension r.qShape[1]?
  return ⟨q,batches,heads⟩

inductive StepResult (S : Type) where
  | ok (state : S)
  | failed (error : String) (state : S)

def StepResult.state {S : Type} : StepResult S → S
  | .ok s | .failed _ s => s

/-- resolve_neg/contiguous/max/item do not change caller tensor coordinates.
The state may still retain newly allocated scratch and other successful prefix
effects. An error is not replaced by the state before preparation began. -/
structure API (S Key Value : Type) where
  step : Stage → S → StepResult S
  observe : S → Key → Value
  caller : Key → Prop
  binds : Stage → S → Prop
  establish : ∀ stage s t, step stage s = .ok t → binds stage t
  preserve : ∀ stage s other, other ≠ stage → binds other s → binds other (step stage s).state
  frame : ∀ stage s key, stage ≠ .deviceSetup → stage ≠ .copyback → caller key →
    observe (step stage s).state key = observe s key

inductive StepsResult (S : Type) where
  | ok (state : S)
  | failed (stage : Stage) (error : String) (state : S)

def StepsResult.state {S : Type} : StepsResult S → S
  | .ok s | .failed _ _ s => s

def steps {S Key Value : Type} (api : API S Key Value) : List Stage → S → StepsResult S
  | [],s => .ok s
  | stage :: rest,s => match api.step stage s with
    | .ok t => steps api rest t
    | .failed e t => .failed stage e t

/-- A concrete successful prefix followed by its first failed API operation. -/
inductive FailedAt {S Key Value : Type} (api : API S Key Value) :
    List Stage → S → Stage → String → S → Prop
  | here {stage rest s e t} (h : api.step stage s = .failed e t) :
      FailedAt api (stage :: rest) s stage e t
  | later {stage rest s u failed e t} (h : api.step stage s = .ok u)
      (tail : FailedAt api rest u failed e t) : FailedAt api (stage :: rest) s failed e t

theorem steps_failure {S Key Value : Type} (api : API S Key Value)
    (jobs : List Stage) (s : S) (stage : Stage) (error : String) (t : S)
    (h : steps api jobs s = .failed stage error t) : FailedAt api jobs s stage error t := by
  induction jobs generalizing s with
  | nil => simp [steps] at h
  | cons next rest ih =>
    cases hs : api.step next s with
    | failed e u =>
      simp only [steps,hs,StepsResult.failed.injEq] at h
      rcases h with ⟨rfl,rfl,rfl⟩
      exact .here hs
    | ok u => exact .later hs (ih u (by simpa only [steps,hs] using h))

theorem steps_binds {S Key Value : Type} (api : API S Key Value)
    (jobs : List Stage) (s t : S) (run : steps api jobs s = .ok t) :
    ∀ stage, stage ∈ jobs ∨ api.binds stage s → api.binds stage t := by
  induction jobs generalizing s with
  | nil => cases run; intro stage hs; exact hs.resolve_left (by simp)
  | cons next rest ih =>
    cases he : api.step next s with
    | failed e u => simp [steps,he] at run
    | ok u =>
      have hr : steps api rest u = .ok t := by simpa [steps,he] using run
      intro stage hs
      apply ih u hr stage
      by_cases eq : stage = next
      · subst stage
        exact Or.inr (api.establish next s u he)
      · rcases hs with hmem | hold
        · exact Or.inl ((List.mem_cons.mp hmem).resolve_left eq)
        · exact Or.inr (by simpa [he] using api.preserve next s stage eq hold)

theorem steps_frame {S Key Value : Type} (api : API S Key Value)
    (jobs : List Stage) (s : S)
    (readOnly : ∀ stage ∈ jobs, stage ≠ .deviceSetup ∧ stage ≠ .copyback)
    (key : Key) (hk : api.caller key) :
    api.observe (steps api jobs s).state key = api.observe s key := by
  induction jobs generalizing s with
  | nil => rfl
  | cons stage rest ih =>
    have hstage := readOnly stage (by simp)
    have hf := api.frame stage s key hstage.1 hstage.2 hk
    have hrest : ∀ stage ∈ rest, stage ≠ .deviceSetup ∧ stage ≠ .copyback :=
      fun other ho => readOnly other (by simp [ho])
    cases he : api.step stage s with
    | ok t => simpa only [steps,he,StepResult.state] using
        (ih t hrest).trans (by simpa [he] using hf)
    | failed e t => simpa [steps,he] using hf

inductive PrefixResult (S : Type) where
  | invalid (error : ValidationError) (state : S)
  | failed (stage : Stage) (error : String) (state : S)
  | ready (dimensions : Dimensions) (state : S)

def PrefixResult.state {S : Type} : PrefixResult S → S
  | .invalid _ s | .failed _ _ s | .ready _ s => s

def runPrefix {S Key Value : Type} (api : API S Key Value) (request : Request) (s : S) : PrefixResult S :=
  match steps api normalization s with
  | .failed stage e t => .failed stage e t
  | .ok normalized => match validate request with
    | .error error => .invalid error normalized
    | .ok dimensions => match steps api (synchronization dimensions.batches) normalized with
      | .failed stage e t => .failed stage e t
      | .ok t => .ready dimensions t

theorem normalization_readOnly : ∀ stage ∈ normalization, stage ≠ .deviceSetup ∧ stage ≠ .copyback := by
  intro stage h
  cases stage <;> simp_all [normalization]

theorem synchronization_readOnly (batches : Nat) :
    ∀ stage ∈ synchronization batches, stage ≠ .deviceSetup ∧ stage ≠ .copyback := by
  intro stage h
  cases stage <;> simp_all [synchronization]

theorem prefix_frame {S Key Value : Type} (api : API S Key Value) (request : Request) (s : S)
    (key : Key) (hk : api.caller key) :
    api.observe (runPrefix api request s).state key = api.observe s key := by
  have hn := steps_frame api normalization s normalization_readOnly key hk
  cases hs : steps api normalization s with
  | failed stage e t => simpa [runPrefix,hs,PrefixResult.state,StepsResult.state] using hn
  | ok normalized =>
    have hn' : api.observe normalized key = api.observe s key := by simpa [hs] using hn
    cases hv : validate request with
    | error error => simpa [runPrefix,hs,hv,PrefixResult.state] using hn'
    | ok dimensions =>
      have hm := steps_frame api (synchronization dimensions.batches) normalized
        (synchronization_readOnly _) key hk
      cases he : steps api (synchronization dimensions.batches) normalized <;>
        simpa [runPrefix,hs,hv,he,PrefixResult.state,StepsResult.state] using hm.trans hn'

theorem prefix_binds {S Key Value : Type} (api : API S Key Value) (request : Request)
    (s t : S) (dimensions : Dimensions) (run : runPrefix api request s = .ready dimensions t) :
    validate request = .ok dimensions ∧
      ∀ stage ∈ normalization ++ synchronization dimensions.batches, api.binds stage t := by
  cases hs : steps api normalization s with
  | failed stage e u => simp [runPrefix,hs] at run
  | ok u =>
    cases hv : validate request with
    | error e => simp [runPrefix,hs,hv] at run
    | ok d =>
      cases he : steps api (synchronization d.batches) u with
      | failed stage e v => simp [runPrefix,hs,hv,he] at run
      | ok v =>
        simp only [runPrefix,hs,hv,he,PrefixResult.ready.injEq] at run
        rcases run with ⟨rfl,rfl⟩
        refine ⟨rfl,?_⟩
        intro stage hstage
        apply steps_binds api _ u v he stage
        rcases List.mem_append.mp hstage with hn | hm
        · exact Or.inr (steps_binds api _ s u hs stage (Or.inl hn))
        · exact Or.inl hm

/-- Each error is the first reached source operation, with its completed
normalization/synchronization prefix and actual state retained. -/
theorem prefix_protocol {S Key Value : Type} (api : API S Key Value) (request : Request) (s : S) :
    match runPrefix api request s with
    | .invalid error t => steps api normalization s = .ok t ∧ validate request = .error error
    | .failed stage error t => FailedAt api normalization s stage error t ∨
        ∃ u dimensions, steps api normalization s = .ok u ∧ validate request = .ok dimensions ∧
          FailedAt api (synchronization dimensions.batches) u stage error t
    | .ready dimensions t => validate request = .ok dimensions ∧
        ∃ u, steps api normalization s = .ok u ∧
          steps api (synchronization dimensions.batches) u = .ok t := by
  cases hs : steps api normalization s with
  | failed stage e t =>
    simp only [runPrefix,hs]
    exact Or.inl (steps_failure api _ s stage e t hs)
  | ok u =>
    cases hv : validate request with
    | error e => simp [runPrefix,hs,hv]
    | ok d =>
      cases he : steps api (synchronization d.batches) u with
      | failed stage e t =>
        simp only [runPrefix,hs,hv,he]
        exact Or.inr ⟨u,d,rfl,rfl,steps_failure api _ u stage e t he⟩
      | ok t => simp [runPrefix,hs,hv,he]

#print axioms prefix_frame
#print axioms prefix_binds
#print axioms prefix_protocol
end VeriTile.Bench.Repairs.ContextAttnNopad.Wrapper
