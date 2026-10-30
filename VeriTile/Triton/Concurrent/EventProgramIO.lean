import VeriTile.Triton.Concurrent.Controller
import VeriTile.Meta.StatementAudit

/-! Finite named IO for the closed concurrent event executor.

`decoded` is actual closed syntax from a named source decoder, not a runner.
Task adapters must prove exact source-step/run refinement separately and expose
the source/decoder connection in their public IO. The shared judgment requires
successful decoding, execution to completion, safety of that entire event trace,
finite observed outputs and exact exterior frames. Read-only source accesses
must be retained as events even when no corresponding store is enabled.
-/
namespace VeriTile.Triton.Concurrent
noncomputable section

/-- A finite named strided element window. Its static mask is geometry data,
independent of execution state and prospective results. Signed addresses keep
negative accesses observable to the shared safety checker. -/
structure Window where
  name : String
  region : Nat
  start : Int := 0
  count : Nat
  stride : Int := 1
  active : Fin count → Bool := fun _=>true

def Window.address (w : Window) (i : Fin w.count) : Int := w.start+(i.val:Int)*w.stride

def Window.Contains (w : Window) (region : Nat) (address : Int) : Prop :=
  ∃ i,w.active i=true ∧ region=w.region ∧ address=w.address i

def Window.InBounds (w : Window) (sizes : Nat → Nat) : Prop :=
  ∀ i,w.active i=true → InWindow (sizes w.region) (w.address i)

inductive ScalarSpace where
  | input
  | mem
  deriving DecidableEq, Repr

structure ScalarInput where
  space : ScalarSpace
  window : Window

def WindowValues (windows : List Window) (A : Type) : Type :=
  (port : Fin windows.length) → Fin windows[port].count → A

structure InputValues (scalars : List ScalarInput) (words : List Window) (A : Type) where
  scalars : (port : Fin scalars.length) → Fin scalars[port].window.count → A
  words : WindowValues words (BitVec 32)

structure OutputValues (scalars words : List Window) (A : Type) where
  scalars : WindowValues scalars A
  words : WindowValues words (BitVec 32)

/-- Only closed event tapes or closed controllers can be executed. -/
inductive Program where
  | tape (events : List Event)
  | controller (control : Controller)

/-- Each controller launch begins with source accumulator zero and no prior
returned CAS words; memory and immutable scalar inputs are unchanged. -/
def Program.initialValues (b : Backend V) (s : State V) : State V :=
  {s with registers:=fun _ _=>b.zero,returned:=fun _ _=>0}

/-- The schedule is a finite constructive witness. `Done` is a conclusion;
there is no fairness axiom and a prefix is not mistaken for completion. -/
def Program.Runs (b : Backend V) (program : Program)
    (initial : State V) (events : List Event) (final : State V) : Prop :=
  match program with
  | .tape instructions => events=instructions ∧ final=run b instructions initial
  | .controller control =>
    ∃ schedule completed,
      control.run b schedule (control.initial (Program.initialValues b initial))=some (events,completed) ∧
      control.Done completed ∧ final=completed.values

/-- Input and output observation types may differ (for example, input FP32
words and output four-byte vectors). Observers consume one scalar only; they
cannot inspect state, choose execution, supply store values, or assert safety.

The source type accommodates the existing source readers. Its syntax is held
alongside their `Option Program` decoding result. Exact source refinement is a
named task theorem, rather than an arbitrary callback inside this interface. -/
structure EventProgramIO (Source Input Output V : Type) where
  source : Source
  decoded : Option Program
  scalarInputs : List ScalarInput
  wordInputs : List Window
  scalarOutputs : List Window
  wordOutputs : List Window
  scalarScratch : List Window := []
  wordScratch : List Window := []
  encode : Input → V
  observe : (port : Fin scalarOutputs.length) → Fin scalarOutputs[port].count → V → Output

namespace EventProgramIO
variable {Source Input Output V : Type}

abbrev Inputs (io : EventProgramIO Source Input Output V) := InputValues io.scalarInputs io.wordInputs Input
abbrev Outputs (io : EventProgramIO Source Input Output V) := OutputValues io.scalarOutputs io.wordOutputs Output

def InputBinding (io : EventProgramIO Source Input Output V) (values : io.Inputs) (s : State V) : Prop :=
  (∀ port lane,(io.scalarInputs[port]).window.active lane=true →
    (match (io.scalarInputs[port]).space with
    | .input => s.input
    | .mem => s.mem) (io.scalarInputs[port]).window.region ((io.scalarInputs[port]).window.address lane)=
      io.encode (values.scalars port lane)) ∧
  (∀ port lane,(io.wordInputs[port]).active lane=true →
    s.words (io.wordInputs[port]).region ((io.wordInputs[port]).address lane)=values.words port lane)

def InBounds (io : EventProgramIO Source Input Output V) (bounds : Bounds) : Prop :=
  (∀ port∈io.scalarInputs,port.window.InBounds (match port.space with
    | .input => bounds.input | .mem => bounds.mem)) ∧
  (∀ port∈io.wordInputs,port.InBounds bounds.words) ∧
  (∀ port∈io.scalarOutputs,port.InBounds bounds.mem) ∧
  (∀ port∈io.wordOutputs,port.InBounds bounds.words) ∧
  (∀ port∈io.scalarScratch,port.InBounds bounds.mem) ∧
  (∀ port∈io.wordScratch,port.InBounds bounds.words)

def OutputBinding (io : EventProgramIO Source Input Output V) (values : io.Outputs) (s : State V) : Prop :=
  (∀ port lane,(io.scalarOutputs[port]).active lane=true →
    io.observe port lane (s.mem (io.scalarOutputs[port]).region ((io.scalarOutputs[port]).address lane))=
      values.scalars port lane) ∧
  (∀ port lane,(io.wordOutputs[port]).active lane=true →
    s.words (io.wordOutputs[port]).region ((io.wordOutputs[port]).address lane)=values.words port lane)

def ScalarWritten (io : EventProgramIO Source Input Output V) (region : Nat) (address : Int) : Prop :=
  ∃ window∈io.scalarOutputs++io.scalarScratch,window.Contains region address

def WordWritten (io : EventProgramIO Source Input Output V) (region : Nat) (address : Int) : Prop :=
  ∃ window∈io.wordOutputs++io.wordScratch,window.Contains region address

/-- All immutable inputs, and every scalar/word cell outside the declared
output and private-scratch windows, retain their exact initial values. The
exterior remains arbitrary; it is not imported into a canonical zero heap. -/
def Frame (io : EventProgramIO Source Input Output V) (initial final : State V) : Prop :=
  final.input=initial.input ∧
  (∀ region address,¬io.ScalarWritten region address → final.mem region address=initial.mem region address) ∧
  (∀ region address,¬io.WordWritten region address → final.words region address=initial.words region address)

/-- The same completed source execution supplies safety, observed outputs and
the exact frame. Neither source decode success nor safe execution is an input
admission condition. The formula depends only on the finite logical inputs. -/
def ImplementsOn (io : EventProgramIO Source Input Output V) (b : Backend V)
    (domain : io.Inputs → Prop) (formula : io.Inputs → io.Outputs) : Prop :=
  ∀ values,domain values → ∀ initial,io.InputBinding values initial →
    ∃ program,io.decoded=some program ∧ ∃ events final,
      program.Runs b initial events final ∧
      (∀ bounds,io.InBounds bounds → TapeSafe bounds events) ∧
      io.OutputBinding (formula values) final ∧ io.Frame initial final

def Implements (io : EventProgramIO Source Input Output V) (b : Backend V)
    (formula : io.Inputs → io.Outputs) : Prop := io.ImplementsOn b (fun _=>True) formula

end EventProgramIO

/-- Access-only source loads remain safety obligations even though they do
not update memory or produce an output event. -/
theorem negative_readOnly_unsafe (bounds : Bounds) (region : Nat) :
    ¬Event.Safe bounds (.readOnly (.load region (-1))) := by
  simp [Event.Safe,Expr.Safe,InWindow]

theorem empty_word_window_cas_unsafe (bounds : Bounds) (region : Nat)
    (empty : bounds.words region=0) (thread destination : Nat) :
    ¬Event.Safe bounds (.compareExchange region 0 1 1 thread destination) := by
  simp [Event.Safe,InWindow,empty]

theorem tape_runs (b : Backend V) (events : List Event) (s : State V) :
    Program.Runs b (.tape events) s events (run b events s) := ⟨rfl,rfl⟩

#axiomsClean negative_readOnly_unsafe
#axiomsClean empty_word_window_cas_unsafe
#axiomsClean tape_runs
end
end VeriTile.Triton.Concurrent
