import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeWhole

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Physical
open CrossEntropy1.Host (World Step)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Both API success and failure carry their actual final state. -/
def outcomeState {S A : Type} : Step S A → S
  | .ok _ state => state
  | .error _ state => state

/-- A complete finite native effect trace for an arbitrary wrapper outcome.
It carries only state transitions and byte-write addresses. Its exclusion
certificate covers the entire retained caller allocation, including holes and
partial trailing words. No byte equality or successful outcome is assumed. -/
structure OutcomeFootprint {σ A : Type} (events : ByteEvents σ) (layout : Layout σ)
    (kept : RegionName → Prop) (initial : World Expr σ) (result : Step (World Expr σ) A)
    (outside : RegionName → Nat → Prop) where
  writes : List ByteAddress
  trace : ByteTrace events initial.native (outcomeState result).native writes
  excludes : ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
    offset<(layout.buffer initial.native r).capacity → outside r offset →
    ((layout.buffer initial.native r).device,(layout.buffer initial.native r).base+offset)∉writes

theorem outcome_frame {σ A : Type} {events : ByteEvents σ} {layout : Layout σ}
    {observe selection kept initial result outside}
    (memory : ByteContracts events observe selection)
    (footprint : OutcomeFootprint (A:=A) events layout kept initial result outside) :
    ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
      offset<(layout.buffer initial.native r).capacity → outside r offset →
      observe (outcomeState result).native (layout.buffer initial.native r).device
        ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  intro r held live active offset bound excluded
  exact footprint.trace.frame memory _ _ (footprint.excludes r held live active offset bound excluded)

/-- Forward never grants permission to modify an existing retained caller
byte. This statement includes ordinary allocation/setup/collective errors. -/
theorem forward_all_outcomes {σ : Type} (events : ByteEvents σ) (layout : Layout σ)
    (observe : σ → Nat → Nat → Byte) (selection : Selection σ) (memory : ByteContracts events observe selection)
    (kept : RegionName → Prop) (rt : Runtime Expr σ) (o : Options) (logits labels : Tensor) (initial : World Expr σ)
    (footprint : OutcomeFootprint events layout kept initial (forward algebra rt o logits labels initial) (fun _ _ => True)) :
    ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
      offset<(layout.buffer initial.native r).capacity →
      observe (outcomeState (forward algebra rt o logits labels initial)).native (layout.buffer initial.native r).device
        ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  intro r held live active offset bound
  exact outcome_frame memory footprint r held live active offset bound trivial

/-- An unsuccessful backward may have performed some authorized inplace
stores. All other caller bytes still have a trace-derived frame. Out-of-place
backward permits no writes to previously live caller allocations. -/
theorem backward_all_outcomes {σ : Type} (events : ByteEvents σ) (layout : Layout σ)
    (observe : σ → Nat → Nat → Byte) (selection : Selection σ) (memory : ByteContracts events observe selection)
    (kept : RegionName → Prop) (rt : Runtime Expr σ) (versions : σ → Nat → Nat) (c : Context) (saved : σ)
    (upstream : Tensor) (zUpstream : Option Tensor) (initial : World Expr σ)
    (footprint : OutcomeFootprint events layout kept initial (backwardEntry algebra rt versions c saved upstream zUpstream initial)
      (fun r offset => c.options.inplace≠Bool.true ∨ (OutsideView c.logits r offset ∧ OutsideView c.inplaceLogits r offset))) :
    ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
      offset<(layout.buffer initial.native r).capacity →
      (c.options.inplace≠Bool.true ∨ (OutsideView c.logits r offset ∧ OutsideView c.inplaceLogits r offset)) →
      observe (outcomeState (backwardEntry algebra rt versions c saved upstream zUpstream initial)).native
        (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) :=
  outcome_frame memory footprint

#axiomsClean outcome_frame
#axiomsClean forward_all_outcomes
#axiomsClean backward_all_outcomes
#specNonCircular ByteContracts avoiding [Host.forward,Host.backward,Grid.forward,Grid.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
