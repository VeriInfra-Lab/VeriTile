import VeriTile.Meta.StatementAudit
import Mathlib.Data.List.Basic

/-! LayerNorm wrapper effects. A failed call retains its post-call state, and
the interpreter never replaces an error state with the original state. The
only abstract effects here are individual host API operations. Kernel execution
is deliberately absent from this interface. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper

inductive Role where
  | x | weight | bias | residual | x1 | weight1 | bias1 | rowscale
  | y | y1 | residualOut | mean | rstd | seeds | mask
  deriving DecidableEq, Repr

inductive Stage where
  | residualDtype | unpackShape | shape (role : Role) | stride (role : Role) (axis : Int)
  | assertStride (role : Role) | assertShape (role : Role) | assertNoRowscale
  | contiguous | assertContiguous | saveOriginal
  | resolve (role : Role) | transfer (role : Role) | allocate (role : Role)
  | elementSize | nextPower | width | enterDevice | selectKernel
  | admission | execute | exitDevice | splitMask | returnTuple
  deriving DecidableEq, Repr

inductive Failure where
  | assertion (stage : Stage)
  | rank
  | index (role : Role) (axis : Int)
  | width
  | api (stage : Stage) (code : Nat)
  deriving DecidableEq, Repr

structure World (S : Type) (Key : Type) (Value : Type) where
  observe : S → Key → Value
  caller : Key → Prop

structure Outcome (S : Type) (α : Type) where
  state : S
  result : Except Failure α
  trace : List Stage

/-- Every host step preserves cells present before the wrapper call. New
allocations and other completed-prefix effects remain in `state`. -/
structure Action {S : Type} {Key : Type} {Value : Type}
    (world : World S Key Value) (α : Type) where
  run : S → Outcome S α
  frame : ∀ s key, world.caller key → world.observe (run s).state key = world.observe s key

variable {S : Type} {Key : Type} {Value : Type} {world : World S Key Value}

def Action.pure {α : Type} (value : α) : Action world α where
  run s := ⟨s,.ok value,[]⟩
  frame _ _ _ := rfl

def Action.bind {α β : Type} (a : Action world α) (f : α → Action world β) : Action world β where
  run s :=
    let first := a.run s
    match first.result with
    | .error e => ⟨first.state,.error e,first.trace⟩
    | .ok x => let second := (f x).run first.state
               ⟨second.state,second.result,first.trace ++ second.trace⟩
  frame s key hk := by
    cases h : (a.run s).result with
    | error e => simpa [h] using a.frame s key hk
    | ok x => simpa [h] using ((f x).frame (a.run s).state key hk).trans (a.frame s key hk)

instance : Monad (Action world) where
  pure := Action.pure
  bind := Action.bind

/-- One local API result, including the actual state on its failure path. -/
structure Local (world : World S Key Value) (α : Type) where
  run : S → S × Except Nat α
  frame : ∀ s key, world.caller key → world.observe (run s).1 key = world.observe s key

def call {α : Type} (stage : Stage) (api : Local world α) : Action world α where
  run s := ⟨(api.run s).1, ((api.run s).2).mapError (.api stage), [stage]⟩
  frame := api.frame

def note (stage : Stage) : Action world Unit where
  run s := ⟨s,.ok (),[stage]⟩
  frame _ _ _ := rfl

def check (stage : Stage) (condition : Prop) [Decidable condition] (error : Failure) : Action world Unit where
  run s := ⟨s,if condition then .ok () else .error error,[stage]⟩
  frame _ _ _ := rfl

def checked {α : Type} (stage : Stage) (result : Except Failure α) : Action world α where
  run s := ⟨s,result,[stage]⟩
  frame _ _ _ := rfl

theorem stop_on_error {α β : Type} (a : Action world α) (f : α → Action world β)
    (s : S) (e : Failure) (h : (a.run s).result=.error e) :
    (a.bind f).run s=⟨(a.run s).state,.error e,(a.run s).trace⟩ := by
  simp [Action.bind,h]

theorem bind_success {α β : Type} (a : Action world α) (f : α → Action world β)
    (s : S) (value : β) (h : ((a.bind f).run s).result=.ok value) :
    ∃ first, (a.run s).result=.ok first ∧
      ((f first).run (a.run s).state).result=.ok value ∧
      ((a.bind f).run s).state=((f first).run (a.run s).state).state ∧
      ((a.bind f).run s).trace=(a.run s).trace++((f first).run (a.run s).state).trace := by
  cases he : (a.run s).result with
  | error e => simp [Action.bind,he] at h
  | ok x => exact ⟨x,rfl,by simpa [Action.bind,he] using h,by simp [Action.bind,he],by simp [Action.bind,he]⟩

theorem bind_failure {α β : Type} (a : Action world α) (f : α → Action world β)
    (s : S) (e : Failure) (h : ((a.bind f).run s).result=.error e) :
    ((a.run s).result=.error e ∧ ((a.bind f).run s).state=(a.run s).state ∧
      ((a.bind f).run s).trace=(a.run s).trace) ∨
    ∃ first, (a.run s).result=.ok first ∧
      ((f first).run (a.run s).state).result=.error e ∧
      ((a.bind f).run s).state=((f first).run (a.run s).state).state ∧
      ((a.bind f).run s).trace=(a.run s).trace++((f first).run (a.run s).state).trace := by
  cases he : (a.run s).result with
  | error err =>
    have hh : err=e := by simpa [Action.bind,he] using h
    subst err
    exact Or.inl ⟨rfl,by simp [Action.bind,he],by simp [Action.bind,he]⟩
  | ok x => exact Or.inr ⟨x,rfl,by simpa [Action.bind,he] using h,by simp [Action.bind,he],by simp [Action.bind,he]⟩

/-- Python's device context runs its exit operation even when argument setup
or dispatch fails. A failing exit replaces the pending exception. -/
def inDevice {α : Type} (enter leave : Local world Unit) (body : Action world α) : Action world α :=
  (call .enterDevice enter).bind fun _ =>
  { run := fun s =>
      let first := body.run s
      let last := (call .exitDevice leave).run first.state
      ⟨last.state, match last.result with | .error e => .error e | .ok _ => first.result,
        first.trace++last.trace⟩
    frame := fun s key hk => (leave.frame (body.run s).state key hk).trans (body.frame s key hk) }

theorem enter_failure {α : Type} (enter leave : Local world Unit) (body : Action world α)
    (s : S) (code : Nat) (h : (enter.run s).2=.error code) :
    (inDevice enter leave body).run s=⟨(enter.run s).1,.error (.api .enterDevice code),[.enterDevice]⟩ := by
  simp [inDevice,Action.bind,call,h,Except.mapError]

theorem exit_after_body_failure {α : Type} (enter leave : Local world Unit) (body : Action world α)
    (s : S) (henter : (enter.run s).2=.ok ()) (e : Failure)
    (hbody : (body.run (enter.run s).1).result=.error e) :
    ((inDevice enter leave body).run s).trace=
      [.enterDevice]++(body.run (enter.run s).1).trace++[.exitDevice] ∧
    ((inDevice enter leave body).run s).state=(leave.run (body.run (enter.run s).1).state).1 ∧
    ((inDevice enter leave body).run s).result=
      match (leave.run (body.run (enter.run s).1).state).2 with
      | .ok _ => .error e | .error code => .error (.api .exitDevice code) := by
  cases hl : (leave.run (body.run (enter.run s).1).state).2 <;>
    simp [inDevice,Action.bind,call,henter,hbody,hl,Except.mapError]

#axiomsClean stop_on_error
#axiomsClean bind_failure
#axiomsClean enter_failure
#axiomsClean exit_after_body_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
