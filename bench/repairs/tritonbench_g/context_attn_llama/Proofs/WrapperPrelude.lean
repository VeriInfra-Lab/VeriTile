import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeGraph

/-! The actual ordered wrapper prefix. The generic first-failure/frame proof
pattern is reused from the checked Nopad wrapper, with Llama's own operations:
request/start/sequence/prompt normalization, page normalization, output clone,
shape assertions and three ordered tolist snapshots. There is no max.item.
Local API contracts concern individual tensor operations, never kernel success.
-/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section

inductive Stage where
  | resolveQ | resolveK | resolveV
  | resolveRequest | contiguousRequest | resolveStart | contiguousStart
  | resolveSequence | contiguousSequence | resolvePrompt | contiguousPrompt
  | resolvePages | resolveOut | cloneOut
  | tolistStart | tolistSequence | tolistPrompt | selectKernel
  | qStride0 | qStride1 | qStride2 | kStride0 | kStride1 | kStride2
  | vStride0 | vStride1 | vStride2 | outStride0 | outStride1 | outStride2
  | pagesStride0 | pagesStride1 | launchAdmission
  deriving DecidableEq

def normalization : List Stage :=
  [.resolveQ,.resolveK,.resolveV,.resolveRequest,.contiguousRequest,
   .resolveStart,.contiguousStart,.resolveSequence,.contiguousSequence,
   .resolvePrompt,.contiguousPrompt,.resolvePages,.resolveOut,.cloneOut]

def setup : List Stage :=
  [.tolistStart,.tolistSequence,.tolistPrompt,.selectKernel,
   .qStride0,.qStride1,.qStride2,.kStride0,.kStride1,.kStride2,
   .vStride0,.vStride1,.vStride2,.outStride0,.outStride1,.outStride2,
   .pagesStride0,.pagesStride1,.launchAdmission]

structure Request where
  qShape : List Nat
  kShape : List Nat
  vShape : List Nat
  sequenceShape : List Nat
  maxInputLen : Nat
  tesla : Bool

inductive ValidationError where
  | queryLastDimension | keyLastDimension | valueLastDimension
  | equalDimensions | supportedDimension | metadataBatchDimension
  | queryHeadDimension | keyHeadDimension | zeroKeyHeads
  deriving DecidableEq

structure Dimensions where
  width : Nat
  batches : Nat
  heads : Nat
  kvGroup : Nat
  deriving DecidableEq

def lookup (error : ValidationError) : Option Nat → Except ValidationError Nat
  | none => .error error | some value => .ok value

def validate (r : Request) : Except ValidationError Dimensions := do
  let q ← lookup .queryLastDimension r.qShape.getLast?
  let k ← lookup .keyLastDimension r.kShape.getLast?
  let v ← lookup .valueLastDimension r.vShape.getLast?
  if ¬ (q = k ∧ k = v) then throw .equalDimensions
  if ¬ (k = 16 ∨ k = 32 ∨ k = 64 ∨ k = 128 ∨ k = 256) then throw .supportedDimension
  let batches ← lookup .metadataBatchDimension r.sequenceShape[0]?
  let heads ← lookup .queryHeadDimension r.qShape[1]?
  let kh ← lookup .keyHeadDimension r.kShape[1]?
  if kh = 0 then throw .zeroKeyHeads
  return ⟨q,batches,heads,heads/kh⟩

inductive StepResult (S : Type) where
  | ok (state : S)
  | failed (error : String) (state : S)

def StepResult.state {S : Type} : StepResult S → S
  | .ok s | .failed _ s => s

/-- resolve_neg/contiguous/clone/snapshot/descriptor setup preserve caller cells.
The state may still retain newly allocated scratch and other successful prefix
effects. An error is not replaced by the state before preparation began. -/
structure API (S Key Value : Type) where
  step : Stage → S → StepResult S
  observe : S → Key → Value
  caller : Key → Prop
  binds : Stage → S → Prop
  establish : ∀ stage s t, step stage s = .ok t → binds stage t
  preserve : ∀ stage s other, other ≠ stage → binds other s → binds other (step stage s).state
  frame : ∀ stage s key, caller key →
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
    (key : Key) (hk : api.caller key) :
    api.observe (steps api jobs s).state key = api.observe s key := by
  induction jobs generalizing s with
  | nil => rfl
  | cons stage rest ih =>
    have hf := api.frame stage s key hk
    cases he : api.step stage s with
    | ok t => simpa only [steps,he,StepResult.state] using
        (ih t).trans (by simpa [he] using hf)
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
    | .ok dimensions => match steps api setup normalized with
      | .failed stage e t => .failed stage e t
      | .ok t => .ready dimensions t

theorem prefix_frame {S Key Value : Type} (api : API S Key Value) (request : Request) (s : S)
    (key : Key) (hk : api.caller key) :
    api.observe (runPrefix api request s).state key = api.observe s key := by
  have hn := steps_frame api normalization s key hk
  cases hs : steps api normalization s with
  | failed stage e t => simpa [runPrefix,hs,PrefixResult.state,StepsResult.state] using hn
  | ok normalized =>
    have hn' : api.observe normalized key = api.observe s key := by simpa [hs] using hn
    cases hv : validate request with
    | error error => simpa [runPrefix,hs,hv,PrefixResult.state] using hn'
    | ok dimensions =>
      have hm := steps_frame api setup normalized key hk
      cases he : steps api setup normalized <;>
        simpa [runPrefix,hs,hv,he,PrefixResult.state,StepsResult.state] using hm.trans hn'

theorem prefix_binds {S Key Value : Type} (api : API S Key Value) (request : Request)
    (s t : S) (dimensions : Dimensions) (run : runPrefix api request s = .ready dimensions t) :
    validate request = .ok dimensions ∧ ∀ stage ∈ normalization ++ setup, api.binds stage t := by
  cases hs : steps api normalization s with
  | failed stage e u => simp [runPrefix,hs] at run
  | ok u =>
    cases hv : validate request with
    | error e => simp [runPrefix,hs,hv] at run
    | ok d =>
      cases he : steps api setup u with
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

/-- Failure retains the actual successful-prefix state and the first failed
operation; validation is reached only after all normalization and clone calls. -/
theorem prefix_protocol {S Key Value : Type} (api : API S Key Value) (request : Request) (s : S) :
    match runPrefix api request s with
    | .invalid error t => steps api normalization s = .ok t ∧ validate request = .error error
    | .failed stage error t => FailedAt api normalization s stage error t ∨
        ∃ u dimensions, steps api normalization s = .ok u ∧ validate request = .ok dimensions ∧
          FailedAt api setup u stage error t
    | .ready dimensions t => validate request = .ok dimensions ∧
        ∃ u, steps api normalization s = .ok u ∧ steps api setup u = .ok t := by
  cases hs : steps api normalization s with
  | failed stage e t =>
    simp only [runPrefix,hs]
    exact Or.inl (steps_failure api _ s stage e t hs)
  | ok u =>
    cases hv : validate request with
    | error e => simp [runPrefix,hs,hv]
    | ok d =>
      cases he : steps api setup u with
      | failed stage e t =>
        simp only [runPrefix,hs,hv,he]
        exact Or.inr ⟨u,d,rfl,rfl,steps_failure api _ u stage e t he⟩
      | ok t => simp [runPrefix,hs,hv,he]

#print axioms prefix_protocol
#print axioms prefix_frame
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Wrapper
