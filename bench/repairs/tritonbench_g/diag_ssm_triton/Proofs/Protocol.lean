import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ByteMemory

/-! Ordered API outcomes. A failure carries the actual state at that call and
ends the continuation; earlier writes to owned allocations are not rolled back.
The generic composition lemma preserves every caller byte through both success
and failure, once each local operation has its ordinary frame contract. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Protocol
open ByteMemory

inductive Role where
  | initial | input | transition | output | gradOutput
  | gradInitial | gradInput | gradTransition | returnedTransition
  deriving DecidableEq, Repr

inductive Call where
  | contiguousCheck (role : Role)
  | shape (role : Role)
  | resolvedCheck (role : Role)
  | contiguous (role : Role)
  | zerosLike (role : Role)
  | emptyLike (role : Role)
  | dtype
  | complexCheck (role : Role)
  | prepareKernel (backward complex : Bool)
  | realView (role : Role)
  | launch (backward complex : Bool)
  | save
  | restore
  | sumDim0
  deriving DecidableEq, Repr

inductive Error where
  | assertion
  | shape
  | dtype
  | library (code : Nat)
  | savedVersion
  | runtime (code : Nat)
  deriving DecidableEq, Repr

inductive Result (State α : Type) where
  | success (value : α) (state : State) (trace : List Call)
  | failure (error : Error) (state : State) (trace : List Call)

def Result.state {State α : Type} : Result State α → State
  | .success _ state _ => state
  | .failure _ state _ => state

def Result.trace {State α : Type} : Result State α → List Call
  | .success _ _ trace => trace
  | .failure _ _ trace => trace

def Result.prepend {State α : Type} (earlier : List Call) : Result State α → Result State α
  | .success value state trace => .success value state (earlier ++ trace)
  | .failure error state trace => .failure error state (earlier ++ trace)

theorem Result.prepend_trace {State α : Type} (earlier : List Call) (result : Result State α) :
    (result.prepend earlier).trace = earlier ++ result.trace := by cases result <;> rfl

abbrev Program (State α : Type) := State → Result State α

def pure {State α : Type} (value : α) : Program State α := fun state => .success value state []

def bind {State α β : Type} (first : Program State α) (next : α → Program State β) : Program State β :=
  fun state => match first state with
  | .failure error failed trace => .failure error failed trace
  | .success value intermediate trace => (next value intermediate).prepend trace

def raise {State α : Type} (error : Error) : Program State α := fun state => .failure error state []

/-- The call is recorded exactly once even when its local service fails. -/
def call {State α : Type} (label : Call) (action : Program State α) : Program State α :=
  fun state => (action state).prepend [label]

theorem failure_stops {State α β : Type} (first : Program State α) (next : α → Program State β)
    (state failed : State) (error : Error) (trace : List Call)
    (h : first state = .failure error failed trace) :
    bind first next state = .failure error failed trace := by simp [bind, h]

theorem success_continues {State α β : Type} (first : Program State α) (next : α → Program State β)
    (state intermediate : State) (value : α) (trace : List Call)
    (h : first state = .success value intermediate trace) :
    bind first next state = (next value intermediate).prepend trace := by simp [bind, h]

def Preserves {State α : Type} (heap : State → Heap) (caller : Backing) (program : Program State α) : Prop :=
  ∀ state, Frame (heap state) (heap (program state).state) caller

theorem pure_preserves {State α : Type} (heap : State → Heap) (caller : Backing) (value : α) :
    Preserves heap caller (pure value) := by intro state; exact Frame.refl _ _

theorem raise_preserves {State α : Type} (heap : State → Heap) (caller : Backing) (error : Error) :
    Preserves heap caller (raise (α := α) error) := by intro state; exact Frame.refl _ _

theorem call_preserves {State α : Type} (heap : State → Heap) (caller : Backing)
    (label : Call) (action : Program State α) (h : Preserves heap caller action) :
    Preserves heap caller (call label action) := by
  intro state
  have frame := h state
  cases hs : action state <;> simpa [call, hs, Result.prepend, Result.state] using frame

theorem bind_preserves {State α β : Type} (heap : State → Heap) (caller : Backing)
    (first : Program State α) (next : α → Program State β)
    (hfirst : Preserves heap caller first) (hnext : ∀ value, Preserves heap caller (next value)) :
    Preserves heap caller (bind first next) := by
  intro state
  have before := hfirst state
  cases hs : first state with
  | failure error failed trace => simpa [bind, hs, Result.state] using before
  | success value intermediate trace =>
    have before : Frame (heap state) (heap intermediate) caller := by simpa [hs, Result.state] using before
    have after := hnext value intermediate
    have total := before.trans after
    cases hn : next value intermediate <;>
      simpa [bind, hs, hn, Result.prepend, Result.state] using total

def sequence {State : Type} : List (Program State Unit) → Program State Unit
  | [] => pure ()
  | first :: rest => bind first (fun _ => sequence rest)

theorem sequence_preserves {State : Type} (heap : State → Heap) (caller : Backing)
    (programs : List (Program State Unit)) (localFrames : ∀ program ∈ programs, Preserves heap caller program) :
    Preserves heap caller (sequence programs) := by
  induction programs with
  | nil => exact pure_preserves heap caller ()
  | cons first rest ih =>
    apply bind_preserves heap caller first (fun _ => sequence rest)
    · exact localFrames first (by simp)
    · intro _
      exact ih (by intro p hp; exact localFrames p (by simp [hp]))

/-- A stateful list of named ordinary calls. This lemma is about sequencing;
task-specific dataflow appears in the wrapper definition, not in a call list. -/
def namedSequence {State : Type} (service : Call → Program State Unit) (labels : List Call) : Program State Unit :=
  sequence (labels.map (fun label => call label (service label)))

def PrefixTrace (observed expected : List Call) := ∃ tail, expected = observed ++ tail

theorem namedSequence_trace {State : Type} (service : Call → Program State Unit)
    (noNestedTrace : ∀ label state, (service label state).trace = [])
    (labels : List Call) (state : State) :
    PrefixTrace (namedSequence service labels state).trace labels := by
  induction labels generalizing state with
  | nil => exact ⟨[], rfl⟩
  | cons label rest ih =>
    have empty := noNestedTrace label state
    cases hs : service label state with
    | failure error failed trace =>
      have ht : trace = [] := by simpa [hs, Result.trace] using empty
      subst trace
      exact ⟨rest, by simp [namedSequence, sequence, call, hs, Result.prepend, bind, Result.trace]⟩
    | success value intermediate trace =>
      have ht : trace = [] := by simpa [hs, Result.trace] using empty
      subst trace
      rcases ih intermediate with ⟨tail, htail⟩
      refine ⟨tail, ?_⟩
      have execution : namedSequence service (label :: rest) state =
          (namedSequence service rest intermediate).prepend [label] := by
        simp [namedSequence, sequence, call, hs, Result.prepend, bind]
      rw [execution, Result.prepend_trace]
      simpa using congrArg (label :: ·) htail

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Protocol
