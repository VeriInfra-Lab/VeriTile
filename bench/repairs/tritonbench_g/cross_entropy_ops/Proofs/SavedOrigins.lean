import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentAdapter

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton
open CrossEntropy1.Floating
open CrossEntropy1.Host (World)
open CrossEntropy1.Physical (Layout Byte)
noncomputable section

/-- Ordinary immutable storage between source calls. This is a relation over
saved allocation bytes, not a loss/LSE value assertion. A tensor library's
saved-reference/version contract supplies it when retrieval succeeds. -/
structure SavedStorage {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (retained : RegionName → Prop) (saved before : World SignedExpr σ) : Prop where
  wasLive : ∀ r, retained r → before.live r → saved.live r
  buffer : ∀ r, retained r → layout.buffer before.native r=layout.buffer saved.native r
  source : ∀ r i, retained r → before.memory r i=saved.memory r i
  bytes : ∀ r i byte, retained r → wordInside layout before.native r i → rawByte layout observe before.native r i byte=
    rawByte layout observe saved.native r i byte
  rawInput : ∀ r i, ¬retained r → before.live r → wordInside layout before.native r i →
    (before.memory r i).read .real=⟨.raw (rawWordAt layout observe before.native r i),Bool.false⟩

/-- Continuation keeps the prior SSA table and all earlier fusion choices.
It only forgets cells outside retained allocations; those are imported as
new raw caller inputs. It never reconstructs a saved plan from its Expr. -/
structure OriginContinuation {σ : Type} {selection : Selection σ} {fusion readRaw}
    (prior next : OriginTrace selection fusion readRaw) (retained : RegionName → Prop) : Prop where
  policy : next.policy.contract=prior.policy.contract
  nodes : next.initial.nodes=prior.final.nodes
  kept : ∀ r i, retained r → next.initial.cells (r,i)=prior.final.cells (r,i)
  forgotten : ∀ r i, ¬retained r → next.initial.cells (r,i)=none

/-- The forward's proved representation, followed by ordinary saved-storage
preservation, establishes the next call's initial representation. No correct
saved expression or native saved-output value is assumed separately. -/
theorem NumericInitial.from_saved {σ : Type} {selection : Selection σ} {fusion}
    {layout : Layout σ} {observe native prior next saved before exports retained}
    (rep : NumericRep (selection:=selection) (fusion:=fusion) layout observe native prior saved exports)
    (storage : SavedStorage layout observe retained saved before)
    (continuation : OriginContinuation prior next retained) :
    NumericInitial layout observe native next before := by
  have rawSame : ∀ r i, retained r → wordInside layout before.native r i → rawWordAt layout observe before.native r i=
      rawWordAt layout observe saved.native r i := by
    intro r i held backed
    simp only [rawWordAt,storage.bytes r i _ held backed]
  constructor
  · intro r i live backed absent
    by_cases held : retained r
    · have oldBacked : wordInside layout saved.native r i := by
        simpa only [wordInside,storage.buffer r held] using backed
      rw [storage.source r i held,rep.undefined r i (storage.wasLive r held live) oldBacked
        ((continuation.kept r i held).symm.trans absent),rawSame r i held backed]
    · exact storage.rawInput r i held live backed
  · intro r i stored live backed atCell
    have held : retained r := by
      by_contra different
      rw [continuation.forgotten r i different] at atCell
      cases atCell
    obtain ⟨node,nodeAt,cell,bytes⟩ := rep.defined r i stored
      ((continuation.kept r i held).symm.trans atCell) (storage.wasLive r held live)
      (by simpa only [wordInside,storage.buffer r held] using backed)
    refine ⟨node,by rw [continuation.nodes]; exact nodeAt,?_,?_⟩
    · rw [storage.source r i held]
      exact cell
    · intro byte
      rw [storage.bytes r i byte held backed,bytes byte,continuation.policy]

#axiomsClean NumericInitial.from_saved
#specNonCircular SavedStorage avoiding [Spec.loss,Spec.zLoss,Spec.gradient,Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular OriginContinuation avoiding [Spec.loss,Spec.zLoss,Spec.gradient,Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
