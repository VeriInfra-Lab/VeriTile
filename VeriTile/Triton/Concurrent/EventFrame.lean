import VeriTile.Triton.Concurrent.EventProgramIO

/-! Exact exterior frames for the fixed executor. The footprint is derived
from event syntax; neither memory contents nor a supplied safety predicate can
hide a write. A controller's final heaps are those of its actual emitted tape.
-/
namespace VeriTile.Triton.Concurrent
set_option linter.unusedSimpArgs false

def Event.WritesMem (event : Event) (region : Nat) (address : Int) : Prop :=
  match event with
  | .store r a _ | .atomicAdd r a _ => region=r ∧ address=a
  | _ => False

def Event.WritesWord (event : Event) (region : Nat) (address : Int) : Prop :=
  match event with
  | .compareExchange r a _ _ _ _ | .exchange r a _ => region=r ∧ address=a
  | _ => False

namespace Frame

theorem heap_update_outside {A : Type} (heap : Nat → Int → A)
    (targetRegion : Nat) (targetAddress : Int) (value : A)
    (region : Nat) (address : Int) (outside : ¬(region=targetRegion ∧ address=targetAddress)) :
    update heap targetRegion (update (heap targetRegion) targetAddress value) region address=
      heap region address := by
  by_cases sameRegion : region=targetRegion
  · subst region
    have different : address≠targetAddress := fun equal=>outside ⟨rfl,equal⟩
    simp [update,different]
  · simp [update,sameRegion]

theorem execute_input (b : Backend V) (event : Event) (s : State V) :
    (execute b event s).input=s.input := by cases event <;> rfl

theorem execute_mem (b : Backend V) (event : Event) (s : State V)
    (region : Nat) (address : Int) (outside : ¬event.WritesMem region address) :
    (execute b event s).mem region address=s.mem region address := by
  cases event with
  | store r a value => exact heap_update_outside s.mem r a _ region address outside
  | atomicAdd r a value => exact heap_update_outside s.mem r a _ region address outside
  | assign | compareExchange | exchange | readOnly | barrier | silent => rfl

theorem execute_words (b : Backend V) (event : Event) (s : State V)
    (region : Nat) (address : Int) (outside : ¬event.WritesWord region address) :
    (execute b event s).words region address=s.words region address := by
  cases event with
  | compareExchange r a compare replace thread destination =>
    exact heap_update_outside s.words r a _ region address outside
  | exchange r a value => exact heap_update_outside s.words r a _ region address outside
  | store | atomicAdd | assign | readOnly | barrier | silent => rfl

theorem run_input (b : Backend V) (events : List Event) (s : State V) :
    (run b events s).input=s.input := by
  induction events generalizing s with
  | nil => rfl
  | cons event rest ih => exact (ih (execute b event s)).trans (execute_input b event s)

theorem run_mem (b : Backend V) (events : List Event) (s : State V)
    (region : Nat) (address : Int)
    (outside : ∀ event∈events,¬event.WritesMem region address) :
    (run b events s).mem region address=s.mem region address := by
  induction events generalizing s with
  | nil => rfl
  | cons event rest ih =>
    exact (ih (execute b event s) (fun e member=>outside e (by simp [member]))).trans
      (execute_mem b event s region address (outside event (by simp)))

theorem run_words (b : Backend V) (events : List Event) (s : State V)
    (region : Nat) (address : Int)
    (outside : ∀ event∈events,¬event.WritesWord region address) :
    (run b events s).words region address=s.words region address := by
  induction events generalizing s with
  | nil => rfl
  | cons event rest ih =>
    exact (ih (execute b event s) (fun e member=>outside e (by simp [member]))).trans
      (execute_words b event s region address (outside event (by simp)))

end Frame

namespace Controller

theorem step_values (b : Backend V) (program : Controller) (thread : Nat)
    (s next : ControlState V) (event : Event)
    (ran : program.step b thread s=some (event,next)) : next.values=execute b event s.values := by
  cases codeAt : program.code[s.pc thread]? with
  | none => simp [step,codeAt] at ran
  | some code =>
    cases decoding : code.decode s.values thread (s.locals thread) with
    | mk head continuation =>
      simp only [step,codeAt,bind,Option.bind,decoding] at ran
      cases Option.some.inj ran
      rfl

/-- The controller only chooses and emits closed events. All memory effects
come from evaluating that exact tape in order against the current state. -/
theorem run_values (b : Backend V) (program : Controller) (schedule : List Nat)
    (s final : ControlState V) (events : List Event)
    (ran : program.run b schedule s=some (events,final)) :
    final.values=Concurrent.run b events s.values := by
  induction schedule generalizing s final events with
  | nil =>
    simp only [run] at ran
    cases Option.some.inj ran
    rfl
  | cons thread rest ih =>
    by_cases skip : (thread ≥ program.threads || !(program.enabled.eval s.values thread (s.locals thread)))=true
    · simp only [run,skip,ite_true] at ran
      exact ih s final events ran
    · cases stepped : program.step b thread s with
      | none => simp [run,skip,stepped] at ran
      | some pair =>
        obtain ⟨head,next⟩ := pair
        cases resumed : program.run b rest next with
        | none => simp [run,skip,stepped,resumed] at ran
        | some pair =>
          obtain ⟨tail,last⟩ := pair
          simp only [run,skip,ite_false,stepped,bind,Option.bind,resumed] at ran
          obtain ⟨eventsEq,finalEq⟩ := Prod.mk.inj (Option.some.inj ran)
          subst events
          subst final
          rw [ih next last tail resumed,step_values b program thread s next head stepped]
          rfl

end Controller

namespace EventProgramIO

/-- Syntax-derived writes contained in the public output/scratch windows
give the exact frame for every actual successful completed execution. -/
theorem frame_of_writes (io : EventProgramIO Source Input Output V) (b : Backend V)
    (program : Program) (initial final : State V) (events : List Event)
    (ran : program.Runs b initial events final)
    (scalarWrites : ∀ event∈events,∀ region address,
      event.WritesMem region address → io.ScalarWritten region address)
    (wordWrites : ∀ event∈events,∀ region address,
      event.WritesWord region address → io.WordWritten region address) : io.Frame initial final := by
  have traceFrame (start : State V) : io.Frame start (run b events start) := by
    refine ⟨Frame.run_input b events start,?_,?_⟩
    · intro region address outside
      exact Frame.run_mem b events start region address
        (fun event member write=>outside (scalarWrites event member region address write))
    · intro region address outside
      exact Frame.run_words b events start region address
        (fun event member write=>outside (wordWrites event member region address write))
  cases program with
  | tape instructions =>
    obtain ⟨rfl,rfl⟩ := ran
    exact traceFrame initial
  | controller control =>
    obtain ⟨schedule,completed,completedRun,done,rfl⟩ := ran
    rw [Controller.run_values b control schedule _ completed events completedRun]
    exact traceFrame (Program.initialValues b initial)

end EventProgramIO

#axiomsClean Frame.run_input
#axiomsClean Frame.run_mem
#axiomsClean Frame.run_words
#axiomsClean Controller.run_values
#axiomsClean EventProgramIO.frame_of_writes
end VeriTile.Triton.Concurrent
