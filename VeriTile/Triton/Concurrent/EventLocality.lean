import VeriTile.Triton.Concurrent.Controller
import VeriTile.Meta.StatementAudit

/-! Locality of the fixed event/controller executor. Only the bounded portions
of the three input/value/word heaps must agree. Registers and returned CAS words
agree exactly because they influence subsequent expressions and control flow.
Every read, including read-only events, is justified by the actual safe trace. -/
namespace VeriTile.Triton.Concurrent
namespace Locality
set_option linter.unusedSimpArgs false

def HeapAgree {A : Type} (sizes : Nat → Nat) (a b : Nat → Int → A) : Prop :=
  ∀ region address,InWindow (sizes region) address → a region address=b region address

structure Agree (bounds : Bounds) (s t : State V) : Prop where
  input : HeapAgree bounds.input s.input t.input
  mem : HeapAgree bounds.mem s.mem t.mem
  words : HeapAgree bounds.words s.words t.words
  registers : s.registers=t.registers
  returned : s.returned=t.returned

theorem heap_update {A : Type} {sizes : Nat → Nat} {a b : Nat → Int → A}
    (h : HeapAgree sizes a b) (region : Nat) (address : Int) (x y : A) (equal : x=y) :
    HeapAgree sizes (update a region (update (a region) address x))
      (update b region (update (b region) address y)) := by
  intro r i hi
  by_cases hr : r=region
  · subst r
    by_cases ha : i=address
    · simp [update,ha,equal]
    · simpa only [update,ite_true,if_neg ha] using h region i hi
  · simpa only [update,if_neg hr] using h r i hi

theorem expr_eval (b : Backend V) (bounds : Bounds) (s t : State V)
    (h : Agree bounds s t) (e : Expr) (safe : e.Safe bounds) : e.eval b s=e.eval b t := by
  induction e with
  | zero => rfl
  | input c i => exact h.input c i safe
  | load r a => exact h.mem r a safe
  | register p l => simp only [Expr.eval,h.registers]
  | add l r ihl ihr => simp only [Expr.eval,ihl safe.1,ihr safe.2]
  | mma8 l r seed ihl ihr ihs =>
    simp only [Expr.eval]
    congr 1
    · funext i; exact ihl i (safe.1 i)
    · funext i; exact ihr i (safe.2.1 i)
    · exact ihs safe.2.2
  | halfRoundTrip e ih => simp only [Expr.eval,ih safe]

theorem execute_agree (b : Backend V) (bounds : Bounds) (s t : State V)
    (h : Agree bounds s t) (e : Event) (safe : e.Safe bounds) :
    Agree bounds (execute b e s) (execute b e t) := by
  cases e with
  | store r a value =>
    exact ⟨h.input,heap_update h.mem r a _ _ (expr_eval b bounds s t h value safe.2),
      h.words,h.registers,h.returned⟩
  | assign p l value =>
    refine ⟨h.input,h.mem,h.words,?_,h.returned⟩
    simp only [execute,h.registers,expr_eval b bounds s t h value safe]
  | atomicAdd r a value =>
    refine ⟨h.input,heap_update h.mem r a _ _ ?_,h.words,h.registers,h.returned⟩
    rw [h.mem r a safe.1,expr_eval b bounds s t h value safe.2]
  | compareExchange r a compare replace p destination =>
    have old := h.words r a safe
    refine ⟨h.input,h.mem,heap_update h.words r a _ _ ?_,h.registers,?_⟩
    · rw [old]
    · simp only [execute,h.returned,old]
  | exchange r a value =>
    exact ⟨h.input,h.mem,heap_update h.words r a value value rfl,h.registers,h.returned⟩
  | readOnly value => exact h
  | barrier => exact h
  | silent => exact h

theorem run_agree (b : Backend V) (bounds : Bounds) (events : List Event)
    (s t : State V) (h : Agree bounds s t) (safe : TapeSafe bounds events) :
    Agree bounds (run b events s) (run b events t) := by
  induction events generalizing s t with
  | nil => exact h
  | cons e rest ih =>
    exact ih _ _ (execute_agree b bounds s t h e (safe e (by simp)))
      (fun e he=>safe e (by simp [he]))

theorem test_eval (s t : State V) (returned : s.returned=t.returned)
    (thread : Nat) (locals : Nat → Nat) (test : Test) :
    test.eval s thread locals=test.eval t thread locals := by
  induction test <;> simp_all [Test.eval]

theorem next_eval (s t : State V) (returned : s.returned=t.returned)
    (thread : Nat) (locals : Nat → Nat) (next : Next) :
    next.eval s thread locals=next.eval t thread locals := by
  induction next with
  | goto label assignments => rfl
  | branch condition yes no ihy ihn =>
    simp only [Next.eval,test_eval s t returned,ihy,ihn]

theorem code_decode (s t : State V) (returned : s.returned=t.returned)
    (thread : Nat) (locals : Nat → Nat) (code : Code) :
    code.decode s thread locals=code.decode t thread locals := by
  induction code with
  | emit event next => rfl
  | branch condition yes no ihy ihn =>
    simp only [Code.decode,test_eval s t returned,ihy,ihn]

structure ControlAgree (bounds : Bounds) (s t : ControlState V) : Prop where
  values : Agree bounds s.values t.values
  pc : s.pc=t.pc
  locals : s.locals=t.locals

theorem done_iff (bounds : Bounds) (program : Controller) (s t : ControlState V)
    (h : ControlAgree bounds s t) : program.Done s ↔ program.Done t := by
  simp only [Controller.Done,h.pc,h.locals,test_eval s.values t.values h.values.returned]

theorem step_transfer (b : Backend V) (bounds : Bounds) (program : Controller)
    (thread : Nat) (s t next : ControlState V) (event : Event)
    (h : ControlAgree bounds s t) (ran : program.step b thread s=some (event,next))
    (safe : event.Safe bounds) :
    ∃ other,program.step b thread t=some (event,other) ∧ ControlAgree bounds next other := by
  cases codeAt : program.code[s.pc thread]? with
  | none => simp [Controller.step,codeAt] at ran
  | some code =>
    have codeAt' : program.code[t.pc thread]?=some code := by rw [←h.pc]; exact codeAt
    have decoded := code_decode s.values t.values h.values.returned thread (s.locals thread) code
    rw [h.locals] at decoded
    cases decoding : code.decode s.values thread (t.locals thread) with
    | mk ev continuation =>
      have decoding' : code.decode t.values thread (t.locals thread)=(ev,continuation) := decoded.symm.trans decoding
      simp only [Controller.step,codeAt,bind,Option.bind,h.locals,decoding] at ran
      cases Option.some.inj ran
      have values := execute_agree b bounds s.values t.values h.values event safe
      have nextEq := next_eval (execute b event s.values) (execute b event t.values)
        values.returned thread (t.locals thread) continuation
      refine ⟨⟨execute b event t.values,
        update t.pc thread (continuation.eval (execute b event t.values) thread (t.locals thread)).1,
        update t.locals thread (continuation.eval (execute b event t.values) thread (t.locals thread)).2⟩,?_,?_⟩
      · simp only [Controller.step,codeAt',bind,Option.bind,decoding']
      · exact ⟨values,by simp only [nextEq,h.pc],by simp only [nextEq,h.locals]⟩

/-- A safe constructive schedule can be replayed with arbitrary exterior
memory. It emits exactly the same events and preserves bounded agreement. -/
theorem run_transfer (b : Backend V) (bounds : Bounds) (program : Controller)
    (schedule : List Nat) (s t : ControlState V) (events : List Event)
    (final : ControlState V) (h : ControlAgree bounds s t)
    (ran : program.run b schedule s=some (events,final))
    (safe : TapeSafe bounds events) :
    ∃ other,program.run b schedule t=some (events,other) ∧ ControlAgree bounds final other := by
  induction schedule generalizing s t events final with
  | nil =>
    simp only [Controller.run] at ran
    cases Option.some.inj ran
    exact ⟨t,rfl,h⟩
  | cons thread rest ih =>
    have enabledEq : program.enabled.eval s.values thread (s.locals thread)=
        program.enabled.eval t.values thread (t.locals thread) := by
      rw [h.locals]
      exact test_eval s.values t.values h.values.returned thread (t.locals thread) program.enabled
    by_cases skip : (thread ≥ program.threads || !(program.enabled.eval s.values thread (s.locals thread)))=true
    · have skip' : (thread ≥ program.threads || !(program.enabled.eval t.values thread (t.locals thread)))=true := by
        rw [←enabledEq]; exact skip
      simp only [Controller.run,skip,ite_true] at ran
      obtain ⟨other,otherRun,agree⟩ := ih s t events final h ran safe
      exact ⟨other,by simpa only [Controller.run,skip',ite_true] using otherRun,agree⟩
    · have skip' : ¬(thread ≥ program.threads || !(program.enabled.eval t.values thread (t.locals thread)))=true := by
        rw [←enabledEq]; exact skip
      cases stepped : program.step b thread s with
      | none => simp [Controller.run,skip,stepped] at ran
      | some pair =>
        obtain ⟨head,next⟩ := pair
        cases resumed : program.run b rest next with
        | none => simp [Controller.run,skip,stepped,resumed] at ran
        | some pair =>
          obtain ⟨tail,last⟩ := pair
          simp only [Controller.run,skip,ite_false,stepped,bind,Option.bind,resumed] at ran
          obtain ⟨eventsEq,finalEq⟩ := Prod.mk.inj (Option.some.inj ran)
          subst events
          subst final
          obtain ⟨nextOther,otherStep,agreeNext⟩ := step_transfer b bounds program thread
            s t next head h stepped (safe head (by simp))
          obtain ⟨other,otherRun,agree⟩ := ih next nextOther tail last agreeNext resumed
            (fun event member=>safe event (by simp [member]))
          refine ⟨other,?_,agree⟩
          simp [Controller.run,skip',otherStep,otherRun]

theorem initial_agree (bounds : Bounds) (program : Controller) (s t : State V)
    (h : Agree bounds s t) : ControlAgree bounds (program.initial s) (program.initial t) :=
  ⟨h,rfl,rfl⟩

#axiomsClean expr_eval
#axiomsClean execute_agree
#axiomsClean run_agree
#axiomsClean step_transfer
#axiomsClean run_transfer
end Locality
end VeriTile.Triton.Concurrent
