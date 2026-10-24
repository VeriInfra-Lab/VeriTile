import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Binding
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.NativeFacts
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
noncomputable section
set_option maxHeartbeats 3000000

theorem whole_source_projection {α : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (m : Source.Memory α)
    (call : Native.Preparation api embed r m) (roundtrip : ∀ w,project (embed w)=w)
    (k : Bool) (b h time c : Nat) (hb : b<r.batches) (hh : h<r.heads k)
    (ht : time<r.sequence) (hc : c<r.dimension) :
    Native.logicalRead api (Native.entry api embed project op r m call).memory
      (Native.returned (Native.entry api embed project op r m call) k) b h time c=
      project ((grid op.backend r m (Native.launchRows r) (region k) (r.offset k (b*r.sequence+time) h c)).read .fp32) := by
  rw [(Native.whole api embed project op r m call roundtrip).2.2.1 k b h time c hb hh ht hc]
  have ha := row_bound r b time hb ht
  by_cases active : c<2*r.half
  · rw [grid_readback op r m (Native.launchRows r) (Native.launchRows_nodup r) _ h c k
      (Native.launchRows_present r k _ h c ha hh active) hh hc]
  · rw [grid_other op.backend r m (Native.launchRows r) _ h c k hh hc (Or.inr (by omega))]
    simp [scalarResult,active]

/-- Whole typed source execution and its physical returned-memory projection. -/
theorem whole_source {α : Type} [Inhabited α] (api : API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (m : Source.Memory α)
    (call : Native.Preparation api embed r m) (roundtrip : ∀ w,project (embed w)=w) :
    GridExec op.backend r (Native.launchRows r) m (grid op.backend r m (Native.launchRows r)) ∧
    ∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
      Native.logicalRead api (Native.entry api embed project op r m call).memory
        (Native.returned (Native.entry api embed project op r m call) k) b h time c=
        project ((grid op.backend r m (Native.launchRows r) (region k) (r.offset k (b*r.sequence+time) h c)).read .fp32) :=
  ⟨grid_execution _ _ _ _,whole_source_projection api embed project op r m call roundtrip⟩

#axiomsClean whole_source
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
