import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativePhysical
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WrapperFP

namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeWrapper
open VeriTile.Triton FP.Structural
open NativePhysical WrapperFP
noncomputable section

def logicalWords (input : NativeGraph.Inputs) : LogicalInputs Expr :=
  ⟨fun row head dim => .leaf .fp32 (input.query row head dim),
   fun row head dim => .leaf .fp32 (input.key row head dim),
   fun row head dim => .leaf .fp32 (input.value row head dim)⟩

/-- All outcomes use actual source order and retain reached error states.
Whenever preparation succeeds, its local role contracts bind every graph leaf
to the original logical FP32 input and imply the full native finite-index,
backedge and metadata-byte projection. No whole result is assumed. -/
theorem whole_outcomes (plan : Plan) {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} (input : NativeGraph.Inputs)
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types)
    (view : FPHostSource.OutputView Expr c)
    (p : Preparation Expr c r A (logicalWords input) view)
    (launch : Launch.API Expr) (copy : Copyback.API Expr c view)
    (request : Wrapper.Request) (agree : Agrees c request) (s : Context Expr) :
    Contract (backend plan) p launch copy layoutState request s ∧
    (Wrapper.runPrefix p.api request s).state.caller = s.caller ∧
    ∀ d t, Wrapper.runPrefix p.api request s = .ready d t →
      NativePhysical.Prepared c r t.kernel input view s.caller ∧
      (∀ j : HostPlan.Job (FPHostSource.effective c r t.kernel),
        WideNative.Projects (FPHostSource.effective c r t.kernel) r A layoutState types j ∧
        ∀ iter, iter ≤ (HostPlan.global j.2.2).val →
          (NativeProgram.word (iter*128) + NativeProgram.word 128).toInt = ((iter*128+128 : Nat) : Int)) ∧
      (∀ b, b < c.batches →
        (NativeProgram.word b * NativeProgram.word types.start.bytes).toInt = ((b*types.start.bytes : Nat) : Int) ∧
        (NativeProgram.word b * NativeProgram.word types.length.bytes).toInt = ((b*types.length.bytes : Nat) : Int)) := by
  refine ⟨WrapperFP.whole_outcomes (backend plan) hm ha fits p launch copy request agree s,
    prefix_all_frame p request s,?_⟩
  intro d t hp
  have prepared := prefix_prepared p request s t d hp
  have caller := prefix_caller p request s t d hp
  refine ⟨⟨prepared.metadata,prepared.query,prepared.key,prepared.value,?_⟩,?_,
    WideNative.metadata_byte_offsets hm types alloc⟩
  · rw [← caller]
    exact prepared.initialized
  · have he : FPHostSource.effective c r t.kernel = HostPlan.effective c r layoutState := by
      unfold FPHostSource.effective HostPlan.effective
      rw [FPHostSource.metadataMaximum_eq prepared.metadata,WideDomain.metadataMaximum_eq hm]
    rw [he]
    exact WideNative.complete_projection hm ha types fits

#print axioms whole_outcomes
end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativeWrapper
