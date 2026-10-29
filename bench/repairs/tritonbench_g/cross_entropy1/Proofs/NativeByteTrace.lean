import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ActualCalls
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.EntryFrames

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

abbrev ByteAddress := Nat × Nat

/-- Ordinary byte-store/memory-management steps expose their write footprint.
The numerical contract is only the separate one-word store contract. -/
structure ByteEvents (σ : Type) where
  step : Occurrence σ → σ → σ → List ByteAddress → Prop

structure ByteContracts {σ : Type} (events : ByteEvents σ)
    (observe : σ → Nat → Nat → Byte) (selection : Selection σ) : Prop where
  frame : ∀ site before after writes, events.step site before after writes →
    ∀ device address, (device,address)∉writes → observe after device address=observe before device address
  called : ∀ site before after writes, events.step site before after writes → selection.invoked site.call
  states : ∀ site before after writes, events.step site before after writes →
    before=site.call.before ∧ after=site.call.after

/-- Finite actual write trace. There is no whole-call preservation premise;
its frame theorem is an induction over one-step outside-write contracts. -/
inductive ByteTrace {σ : Type} (events : ByteEvents σ) : σ → σ → List ByteAddress → Prop
  | nil (state) : ByteTrace events state state []
  | cons (site before next after writes rest)
      (step : events.step site before next writes) (tail : ByteTrace events next after rest) :
      ByteTrace events before after (writes++rest)

theorem ByteTrace.frame {σ : Type} {events : ByteEvents σ} {observe selection}
    (api : ByteContracts events observe selection) {before after writes}
    (trace : ByteTrace events before after writes) (device address : Nat)
    (outside : (device,address)∉writes) : observe after device address=observe before device address := by
  induction trace with
  | nil => rfl
  | cons site before next after writes rest step tail ih =>
    have split : (device,address)∉writes ∧ (device,address)∉rest := by simpa only [List.mem_append,not_or] using outside
    exact (ih split.2).trans (api.frame site before next writes step device address split.1)

/-- A native byte observation at an actual complete tensor descriptor. -/
def tensorByte {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (state : σ) (tensor : Tensor) (index : Nat) (byte : Fin 4) : Byte :=
  observe state (layout.buffer state tensor.region).device
    ((layout.buffer state tensor.region).base+4*tensor.index index+byte.val)

structure NativeCallerFrames {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (kept : RegionName → Prop) (initial saved before after : World Floating.Expr σ)
    (inplace : Bool) (work output : Tensor) : Prop where
  forward : ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
    offset<(layout.buffer initial.native r).capacity →
    observe saved.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)
  backward : ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity → OutsideView work r offset → OutsideView output r offset →
    observe after.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      observe before.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)
  outOfPlace : inplace≠Bool.true → ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity →
    observe after.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      observe before.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)

/-- This certificate is address-only: actual step lists and exclusion of
retained caller addresses from their write footprints. It has no byte values.
Allocation/lifetime and source store-footprint proofs discharge these clauses;
holes and partial trailing words are included without rounding capacity. -/
structure CallerWriteFootprints {σ : Type} (events : ByteEvents σ) (layout : Layout σ)
    (kept : RegionName → Prop) (initial saved before after : World Floating.Expr σ)
    (inplace : Bool) (work output : Tensor) where
  forwardWrites : List ByteAddress
  backwardWrites : List ByteAddress
  forwardTrace : ByteTrace events initial.native saved.native forwardWrites
  backwardTrace : ByteTrace events before.native after.native backwardWrites
  forwardOutside : ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
    offset<(layout.buffer initial.native r).capacity →
    ((layout.buffer initial.native r).device,(layout.buffer initial.native r).base+offset)∉forwardWrites
  backwardOutside : ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity → OutsideView work r offset → OutsideView output r offset →
    ((layout.buffer before.native r).device,(layout.buffer before.native r).base+offset)∉backwardWrites
  outOfPlaceOutside : inplace≠Bool.true → ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity →
    ((layout.buffer before.native r).device,(layout.buffer before.native r).base+offset)∉backwardWrites

theorem native_caller_frames {σ : Type} {events : ByteEvents σ} {observe selection}
    (api : ByteContracts events observe selection) {layout kept initial saved before after inplace work output}
    (footprints : CallerWriteFootprints events layout kept initial saved before after inplace work output) :
    NativeCallerFrames layout observe kept initial saved before after inplace work output := by
  constructor
  · intro r held live active offset inside
    exact footprints.forwardTrace.frame api _ _ (footprints.forwardOutside r held live active offset inside)
  · intro r held live active offset inside outsideWork outsideOutput
    exact footprints.backwardTrace.frame api _ _ (footprints.backwardOutside r held live active offset inside outsideWork outsideOutput)
  · intro outOfPlace r held live active offset inside
    exact footprints.backwardTrace.frame api _ _ (footprints.outOfPlaceOutside outOfPlace r held live active offset inside)

#axiomsClean ByteTrace.frame
#axiomsClean native_caller_frames
#specNonCircular ByteContracts avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
