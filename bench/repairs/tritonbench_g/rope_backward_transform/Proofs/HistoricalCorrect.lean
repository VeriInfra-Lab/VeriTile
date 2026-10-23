import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Composition
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
noncomputable section

/-- Independent adjoint target for both returned original-layout tensor views. -/
theorem correct (api : API ℝ Store Byte) (r : Request) (m : Source.Memory ℝ)
    (call : Native.Preparation api id r m) (_abi : Native.IndexABI api id r m call) :
    (∀ k,(Native.returned (Native.entry api id id ideal r m call) k)=call.output k) ∧
    (∀ k,(Native.returned (Native.entry api id id ideal r m call) k).shape=[r.batches,r.heads k,r.sequence,r.dimension]) ∧
    (∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
      Native.logicalRead api (Native.entry api id id ideal r m call).memory
        (Native.returned (Native.entry api id id ideal r m call) k) b h time c=
        mathematicalTarget api id r call.before call.input call.cosine call.sine k b h time c) ∧
    (∀ b,call.caller b → Memory.Outside r (fun k => (call.output k).base) (Native.launchRows r) b →
      api.byte (Native.entry api id id ideal r m call).memory b=api.byte call.before b) := by
  obtain ⟨obj,shape,read,frame⟩ := Native.whole api id id ideal r m call (fun _ => rfl)
  refine ⟨obj,shape,?_,frame⟩
  intro k b h time c hb hh ht hc
  rw [read k b h time c hb hh ht hc]
  change scalarResult ideal r m k (b*r.sequence+time) h c=_
  rw [scalar_binding api id ideal r m call k b h time c hb hh ht hc,ideal_logical]
  rfl

#axiomsClean correct
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
