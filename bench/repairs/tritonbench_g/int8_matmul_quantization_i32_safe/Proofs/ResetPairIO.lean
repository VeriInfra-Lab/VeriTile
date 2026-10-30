import VeriTile.Triton.Float.IntegerDotPerAddressProgramIO
import VeriTile.Spec
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ResetPairIO
open VeriTile Triton FP.Structural IntegerDotPerAddressProgramIO
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

/-- The same typed initializer is executed before both full source programs.
Both complete per-address traces, typed observations and exterior frames are
retained, including the reset's own input preservation and frame. -/
def EquivalentResetOnCells {α β γ Event : Type} [Inhabited β]
    (io : IntegerDotPerAddressProgramIO α β γ Event) (candidate : Variant io)
    (domain : io.Inputs → Prop) (initial : io.RawOutputs) : Prop :=
  ∀ vals,domain vals → ∀ base,io.InputCells vals base →
    let start:=io.patchOutputs vals initial base
    io.InputCells vals start ∧ io.OutputCells vals initial start ∧ io.Frame vals base start ∧
    io.ScheduleComplete vals ∧ candidate.toPerAddressIO.ScheduleComplete vals ∧
    io.Programs vals start ∧ candidate.toPerAddressIO.Programs vals start ∧
    io.SourceLocality vals start ∧ candidate.toPerAddressIO.SourceLocality vals start ∧
    io.SourceCoverage vals start ∧ candidate.toPerAddressIO.SourceCoverage vals start ∧
    ∃ left right : io.RawOutputs,
      (∀ key,io.outputActive vals key →
        io.SourceCellTrace vals start initial key (io.order vals key.1 key.2) (initial key) (left key)) ∧
      (∀ key,io.outputActive vals key →
        candidate.toPerAddressIO.SourceCellTrace vals start initial key
          (io.order vals key.1 key.2) (initial key) (right key)) ∧
      OutputsEqual io candidate vals start left right ∧
      io.Frame vals base (io.patchOutputs vals left start) ∧
      io.Frame vals base (io.patchOutputs vals right start)

namespace Variant
variable {α β γ Event : Type} [Inhabited β] {io : IntegerDotPerAddressProgramIO α β γ Event}

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

theorem implementsReset_of_run_safe (candidate : Variant io)
    (observation : candidate.observe=io.observe)
    (run : ∀ vals event s,(io.invocation vals event).run s=(candidate.invocation vals event).run s)
    (safe : ∀ vals event bounds s,
      (io.invocation vals event).Safe bounds s ↔ (candidate.invocation vals event).Safe bounds s)
    {domain : io.Inputs → Prop} {initial : io.RawOutputs} {f : io.Inputs → io.Outputs}
    (source : io.ImplementsResetOnCells domain initial f) :
    candidate.toPerAddressIO.ImplementsResetOnCells domain initial f := by
  intro vals hd base hin
  obtain ⟨inputs,outputs,resetFrame,complete,programs,locality,coverage,final,trace,values,frame⟩ :=
    source vals hd base hin
  refine ⟨inputs,outputs,resetFrame,complete,?_,?_,?_,final,?_,?_,frame⟩
  · intro current event listed
    obtain ⟨hs,t,ran,hf⟩:=programs current event listed
    exact ⟨fun bounds hb=>(safe vals event bounds _).mp (hs bounds hb),
      t,(run vals event _).symm.trans ran,hf⟩
  · intro event listed key active l r same ls rs lrun rrun
    exact locality event listed key active l r same ls rs
      ((run vals event _).trans lrun) ((run vals event _).trans rrun)
  · intro key active event listed missing current t ran
    exact coverage key active event listed missing current t ((run vals event _).trans ran)
  · intro key active
    exact trace_of_run_safe candidate run safe vals _ initial key (trace key active)
  · simpa only [OutputBinding,Variant.toPerAddressIO,observation] using values
end Variant

theorem of_run_safe {α β γ Event : Type} [Inhabited β]
    {io : IntegerDotPerAddressProgramIO α β γ Event} (candidate : Variant io)
    (observation : candidate.observe=io.observe)
    (run : ∀ vals event s,(io.invocation vals event).run s=(candidate.invocation vals event).run s)
    (safe : ∀ vals event bounds s,
      (io.invocation vals event).Safe bounds s ↔ (candidate.invocation vals event).Safe bounds s)
    {domain : io.Inputs → Prop} {initial : io.RawOutputs} {f : io.Inputs → io.Outputs}
    (source : io.ImplementsResetOnCells domain initial f) :
    EquivalentResetOnCells io candidate domain initial := by
  intro vals hd base hin
  obtain ⟨inputs,outputs,resetFrame,lc,lp,ll,lcv,l,lt,lv,lf⟩:=source vals hd base hin
  obtain ⟨_,_,_,rc,rp,rl,rcv,r,rt,rv,rf⟩:=
    (candidate.implementsReset_of_run_safe observation run safe source) vals hd base hin
  refine ⟨inputs,outputs,resetFrame,lc,rc,lp,rp,ll,rl,lcv,rcv,l,r,lt,rt,?_,lf,rf⟩
  intro o j active
  exact (lv o j active).trans (rv o j active).symm

structure Implementation {α β γ Event : Type} (io : IntegerDotPerAddressProgramIO α β γ Event)
    (domain : io.Inputs → Prop) (initial : io.RawOutputs) where
  invocation : io.Inputs → Event → IntegerDotInvocation β
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → γ

namespace Implementation
variable {α β γ Event : Type} {io : IntegerDotPerAddressProgramIO α β γ Event}
  {domain : io.Inputs → Prop} {initial : io.RawOutputs}
def toVariant (p : Implementation io domain initial) : Variant io :=⟨p.invocation,p.observe⟩
def toIO (p : Implementation io domain initial) :=p.toVariant.toPerAddressIO
def original (io : IntegerDotPerAddressProgramIO α β γ Event) (domain : io.Inputs → Prop) (initial : io.RawOutputs) :
    Implementation io domain initial :=⟨io.invocation,io.observe⟩
def ofVariant (candidate : Variant io) (domain : io.Inputs → Prop) (initial : io.RawOutputs) :
    Implementation io domain initial :=⟨candidate.invocation,candidate.observe⟩
def relativeVariant (left right : Implementation io domain initial) : Variant left.toIO :=⟨right.invocation,right.observe⟩
def Equivalent [Inhabited β] (left right : Implementation io domain initial) : Prop :=
  EquivalentResetOnCells left.toIO (relativeVariant left right) domain initial
instance programSyntax [Inhabited β] : Spec.ProgramSyntax (Implementation io domain initial) where
  Statement:=Implementation io domain initial
  Signature:=Unit
  signature:=fun _=>()
  body:=fun p=>[p]
  structural:=some Equivalent
  sameContext:=Eq
theorem of_equivalentResetOnCells [Inhabited β] (candidate : Variant io)
    (proof : EquivalentResetOnCells io candidate domain initial) :
    Spec.Equiv (original io domain initial) (ofVariant candidate domain initial)
      ([] : Spec.Assumptions (Implementation io domain initial)) := by
  exact Spec.FloatingPoint.ofStructural rfl rfl proof
end Implementation

#axiomsClean of_run_safe
#axiomsClean Implementation.of_equivalentResetOnCells
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ResetPairIO
