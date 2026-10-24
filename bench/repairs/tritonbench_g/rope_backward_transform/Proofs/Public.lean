import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.HistoricalFPEquiv
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Guards

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
open Numerics (Word Tagged leaf)
noncomputable section
set_option maxHeartbeats 3000000

def invoke {α : Type} [Inhabited α] (api : API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (m : Source.Memory α) (call : Native.Preparation api embed r m) :=
  if check api embed r m call then some (Native.entry api embed project op r m call) else none

def RealOutcome (api : API ℝ Store Byte) (r : Request) (m : Source.Memory ℝ)
    (call : Native.Preparation api id r m) : Prop :=
  ∃ out,invoke api id id ideal r m call=some out ∧
    GridExec ideal.backend r (Native.launchRows r) m (grid ideal.backend r m (Native.launchRows r)) ∧
    (∀ k,Native.returned out k=call.output k) ∧
    (∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
      Native.logicalRead api out.memory (Native.returned out k) b h time c=
        mathematicalTarget api id r call.before call.input call.cosine call.sine k b h time c) ∧
    (∀ b,call.caller b → Memory.Outside r (fun k => (call.output k).base) (Native.launchRows r) b →
      api.byte out.memory b=api.byte call.before b) ∧ Native.IndexABI api id r m call

theorem whole_real (api : API ℝ Store Byte) (r : Request) (m : Source.Memory ℝ)
    (call : Native.Preparation api id r m) (accepted : check api id r m call=Bool.true) :
    RealOutcome api r m call := by
  have abi := indexABI api id r m call accepted
  obtain ⟨objects,shape,values,frame⟩ := correct api r m call abi
  exact ⟨Native.entry api id id ideal r m call,by simp [invoke,accepted],
    grid_execution _ _ _ _,objects,values,frame,abi⟩

/-- Full actual backward return, word/four-byte values, source execution and
caller backing-byte frame. No reached-node exactness is required for this
native word statement. Odd tails and no-launch paths are included. -/
def FP32Outcome (api : API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (m : Source.Memory Tagged) (call : Native.Preparation api leaf r m) : Prop :=
  ∃ out,invoke api leaf Tagged.word (Numerics.operations p) r m call=some out ∧
    GridExec (Numerics.operations p).backend r (Native.launchRows r) m
      (grid (Numerics.operations p).backend r m (Native.launchRows r)) ∧
    (∀ k,Native.returned out k=call.output k) ∧
    (∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
      let word := machineTarget api p r call.before call.input call.cosine call.sine k b h time c
      Native.logicalRead api out.memory (Native.returned out k) b h time c=word ∧
      ∀ byte : Fin 4,api.byte out.memory (Native.logicalAddress (Native.returned out k) b h time c+byte.val)=
        Numerics.wordByte word byte) ∧
    (∀ b,call.caller b → Memory.Outside r (fun k => (call.output k).base) (Native.launchRows r) b →
      api.byte out.memory b=api.byte call.before b) ∧ Native.IndexABI api leaf r m call

theorem whole_fp32 (api : API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (m : Source.Memory Tagged) (call : Native.Preparation api leaf r m)
    (codec : ∀ w k,api.encode w k=Numerics.wordByte w k)
    (accepted : check api leaf r m call=Bool.true) : FP32Outcome api p r m call := by
  have abi := indexABI api leaf r m call accepted
  obtain ⟨objects,shape,values,frame⟩ := Native.whole api leaf Tagged.word (Numerics.operations p) r m call (fun _ => rfl)
  refine ⟨Native.entry api leaf Tagged.word (Numerics.operations p) r m call,
    by simp [invoke,accepted],grid_execution _ _ _ _,objects,?_,frame,abi⟩
  intro k b h time c hb hh ht hc
  dsimp only
  have word := default_profile_words api p r m call abi k b h time c hb hh ht hc
  refine ⟨word,?_⟩
  intro byte
  rw [api.read_bytes,codec]
  exact congrArg (fun w => Numerics.wordByte w byte) word

/-- The caller set is exactly the union of all original backing allocations,
including prefixes, row gaps in coefficient views, and unused suffix bytes. -/
theorem backing_frame (api : API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (m : Source.Memory Tagged) (call : Native.Preparation api leaf r m)
    (b : Nat) (backing : (∃ k,(call.input k).allocationBase≤b ∧ b<(call.input k).allocationEnd) ∨
      (call.cosine.allocationBase≤b ∧ b<call.cosine.allocationEnd) ∨
      (call.sine.allocationBase≤b ∧ b<call.sine.allocationEnd))
    (outside : Memory.Outside r (fun k => (call.output k).base) (Native.launchRows r) b) :
    api.byte (Native.entry api leaf Tagged.word (Numerics.operations p) r m call).memory b=api.byte call.before b :=
  (Native.whole api leaf Tagged.word (Numerics.operations p) r m call (fun _ => rfl)).2.2.2
    b ((call.callerBinding b).mpr backing) outside

#axiomsClean whole_real
#axiomsClean whole_fp32
#axiomsClean backing_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.Checked
