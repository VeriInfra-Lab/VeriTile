import VeriTile.Triton.Float.IntegerDotPerAddressProgramIO
import VeriTile.Spec
import VeriTile.Meta.StatementAudit

/-! Task-local adapter for two actual contextual integer-dot per-address source implementations on a fixed memory interface.
A variant may change its concrete invocations and observation. Input encoding,
launch catalog, event coverage and each address's independent order remain
fixed. Both sides must establish full source success/safety/locality/coverage,
complete per-address source-cell traces, output equality and exterior frames.
-/
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.PairIO
open VeriTile Triton FP.Structural
open IntegerDotPerAddressProgramIO
noncomputable section

structure Variant {α β γ Event : Type} (io : IntegerDotPerAddressProgramIO α β γ Event) where
  invocation : io.Inputs → Event → IntegerDotInvocation β
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → γ

def Variant.toPerAddressIO {α β γ Event : Type} {io : IntegerDotPerAddressProgramIO α β γ Event}
    (candidate : Variant io) : IntegerDotPerAddressProgramIO α β γ Event :=
  { io with invocation := candidate.invocation, observe := candidate.observe }

def OutputsEqual {α β γ Event : Type} (io : IntegerDotPerAddressProgramIO α β γ Event)
    (candidate : Variant io) (vals : io.Inputs) (base : State β) (left right : io.RawOutputs) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j)
      ((io.patchOutputs vals left base).mem io.outputs[o].buf (io.write o vals j)) =
    mapCell (candidate.observe o j)
      ((io.patchOutputs vals right base).mem io.outputs[o].buf (io.write o vals j))

def EquivalentOnCells {α β γ Event : Type} [Inhabited β]
    (io : IntegerDotPerAddressProgramIO α β γ Event) (candidate : Variant io)
    (domain : io.Inputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ base, io.InputCells vals base →
  ∀ initial : io.RawOutputs, io.OutputCells vals initial base →
    io.ScheduleComplete vals ∧ candidate.toPerAddressIO.ScheduleComplete vals ∧
    io.Programs vals base ∧ candidate.toPerAddressIO.Programs vals base ∧
    io.SourceLocality vals base ∧ candidate.toPerAddressIO.SourceLocality vals base ∧
    io.SourceCoverage vals base ∧ candidate.toPerAddressIO.SourceCoverage vals base ∧
    ∃ left right : io.RawOutputs,
      (∀ key, io.outputActive vals key →
        io.SourceCellTrace vals base initial key (io.order vals key.1 key.2) (initial key) (left key)) ∧
      (∀ key, io.outputActive vals key →
        candidate.toPerAddressIO.SourceCellTrace vals base initial key
          (io.order vals key.1 key.2) (initial key) (right key)) ∧
      OutputsEqual io candidate vals base left right ∧
      io.Frame vals base (io.patchOutputs vals left base) ∧
      io.Frame vals base (io.patchOutputs vals right base)

def EquivalentCells {α β γ Event : Type} [Inhabited β]
    (io : IntegerDotPerAddressProgramIO α β γ Event) (candidate : Variant io) : Prop :=
  EquivalentOnCells io candidate (fun _ => True)

/-- Complete implementations of one independent target give an actual pair,
retaining both source traces and every address's original event order. -/
theorem EquivalentOnCells.of_implementsOnCells {α β γ Event : Type} [Inhabited β]
    {io : IntegerDotPerAddressProgramIO α β γ Event} {candidate : Variant io}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (left : io.ImplementsOnCells domain f)
    (right : candidate.toPerAddressIO.ImplementsOnCells domain f) :
    EquivalentOnCells io candidate domain := by
  intro vals hd base hin initial hout
  obtain ⟨lcomplete,lprograms,llocal,lcover,l,ltrace,lout,lframe⟩ := left vals hd base hin initial hout
  obtain ⟨rcomplete,rprograms,rlocal,rcover,r,rtrace,rout,rframe⟩ := right vals hd base hin initial hout
  refine ⟨lcomplete,rcomplete,lprograms,rprograms,llocal,rlocal,lcover,rcover,
    l,r,ltrace,rtrace,?_,lframe,rframe⟩
  intro o j active
  exact (lout o j active).trans (rout o j active).symm

namespace Variant
variable {α β γ Event : Type} [Inhabited β] {io : IntegerDotPerAddressProgramIO α β γ Event}

/-- Transport an actual cell trace through equal complete invocation runs and
safe traces. No numeric or output-equality callback is part of a variant. -/
theorem trace_of_run_safe (candidate : Variant io)
    (run : ∀ vals event s, (io.invocation vals event).run s = (candidate.invocation vals event).run s)
    (safe : ∀ vals event bounds s,
      (io.invocation vals event).Safe bounds s ↔ (candidate.invocation vals event).Safe bounds s)
    (vals : io.Inputs) (base : State β) (initial : io.RawOutputs) (key : io.OutputKey)
    {events : List Event} {before after : Value β (io.outputs[key.1]).dtype}
    (trace : io.SourceCellTrace vals base initial key events before after) :
    candidate.toPerAddressIO.SourceCellTrace vals base initial key events before after := by
  induction trace with
  | nil value => exact SourceCellTrace.nil (io := candidate.toPerAddressIO) value
  | cons event rest before middle after final listed ran cell hs frame tail ih =>
    refine SourceCellTrace.cons (io := candidate.toPerAddressIO)
      event rest before middle after final listed ?_ cell ?_ frame ih
    · exact (run vals event _).symm.trans ran
    · intro bounds hb
      exact (safe vals event bounds _).mp (hs bounds hb)

/-- A proved invocation equality transports all existing full-IO obligations.
Every CTA, including one with no active output lane, still has to execute.
Only the output observer must also be identical for this exact-state helper. -/
theorem implements_of_run_safe (candidate : Variant io)
    (observation : candidate.observe=io.observe)
    (run : ∀ vals event s, (io.invocation vals event).run s = (candidate.invocation vals event).run s)
    (safe : ∀ vals event bounds s,
      (io.invocation vals event).Safe bounds s ↔ (candidate.invocation vals event).Safe bounds s)
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (source : io.ImplementsOnCells domain f) :
    candidate.toPerAddressIO.ImplementsOnCells domain f := by
  intro vals hd base hin initial hout
  obtain ⟨complete,programs,locality,coverage,final,trace,values,frame⟩ := source vals hd base hin initial hout
  refine ⟨complete,?_,?_,?_,final,?_,?_,frame⟩
  · intro current event listed
    obtain ⟨hs,t,ran,hf⟩ := programs current event listed
    refine ⟨?_,t,(run vals event _).symm.trans ran,hf⟩
    intro bounds hb
    exact (safe vals event bounds _).mp (hs bounds hb)
  · intro event listed key active l r same ls rs lrun rrun
    exact locality event listed key active l r same ls rs
      ((run vals event _).trans lrun) ((run vals event _).trans rrun)
  · intro key active event listed missing current t ran
    exact coverage key active event listed missing current t ((run vals event _).trans ran)
  · intro key active
    exact trace_of_run_safe candidate run safe vals base initial key (trace key active)
  · simpa only [OutputBinding,Variant.toPerAddressIO,observation] using values

end Variant

theorem EquivalentOnCells.of_run_safe {α β γ Event : Type} [Inhabited β]
    {io : IntegerDotPerAddressProgramIO α β γ Event} (candidate : Variant io)
    (observation : candidate.observe=io.observe)
    (run : ∀ vals event s, (io.invocation vals event).run s = (candidate.invocation vals event).run s)
    (safe : ∀ vals event bounds s,
      (io.invocation vals event).Safe bounds s ↔ (candidate.invocation vals event).Safe bounds s)
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (source : io.ImplementsOnCells domain f) : EquivalentOnCells io candidate domain :=
  .of_implementsOnCells source (candidate.implements_of_run_safe observation run safe source)


/-- Actual source programs of the same fixed public integer-dot IO interface. -/
structure Implementation {α β γ Event : Type} (io : IntegerDotPerAddressProgramIO α β γ Event)
    (domain : io.Inputs → Prop) where
  invocation : io.Inputs → Event → IntegerDotInvocation β
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → γ

namespace Implementation
variable {α β γ Event : Type} {io : IntegerDotPerAddressProgramIO α β γ Event}
  {domain : io.Inputs → Prop}

def toVariant (p : Implementation io domain) : Variant io where
  invocation := p.invocation
  observe := p.observe

def toIO (p : Implementation io domain) := p.toVariant.toPerAddressIO

def original (io : IntegerDotPerAddressProgramIO α β γ Event) (domain : io.Inputs → Prop) :
    Implementation io domain where
  invocation := io.invocation
  observe := io.observe

def ofVariant (candidate : Variant io) (domain : io.Inputs → Prop) : Implementation io domain where
  invocation := candidate.invocation
  observe := candidate.observe

def relativeVariant (left right : Implementation io domain) : Variant left.toIO where
  invocation := right.invocation
  observe := right.observe

def Equivalent [Inhabited β] (left right : Implementation io domain) : Prop :=
  EquivalentOnCells left.toIO (relativeVariant left right) domain

instance programSyntax [Inhabited β] : VeriTile.Spec.ProgramSyntax (Implementation io domain) where
  Statement := Implementation io domain
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some Equivalent
  sameContext := Eq

theorem of_equivalentOnCells [Inhabited β] (candidate : Variant io)
    (proof : EquivalentOnCells io candidate domain) :
    VeriTile.Spec.Equiv (original io domain) (ofVariant candidate domain)
      ([] : VeriTile.Spec.Assumptions (Implementation io domain)) := by
  exact VeriTile.Spec.FloatingPoint.ofStructural rfl rfl proof
end Implementation

#axiomsClean EquivalentOnCells.of_run_safe
#axiomsClean Implementation.of_equivalentOnCells
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.PairIO
