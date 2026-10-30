import bench.tritonbench_g.fused_rotary_embedding.Proofs.IOBounds
import bench.tritonbench_g.fused_rotary_embedding.Proofs.IOResults

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

/-- The shared IO judgment includes actual source safety, complete launch
execution, typed logical outputs, and the exact exterior-cell frame. -/
theorem implements [Inhabited α] [Inhabited β] (rest : Algebra β) (ops : Ops β)
    (encode : α → β) (observe : Kind → β → α) (c : Config) (hc : c.Valid)
    (f : ChannelValues α (inputChannels c.shape) → Kind → Nat → Nat → Nat → α)
    (lower : ∀ xs w, c.shape.Active w → ∀ high,
      observe w.kind (component (pairValues c.shape (fun _=>ops) w
        (fun r a=>encode (inputMemory c.shape xs r a))) high) =
          f xs w.kind w.token w.head (c.shape.dim w high)) :
    (io rest ops encode observe c).ImplementsOn (metadataDomain c)
      (fun xs=>outputFormula c.shape (f xs)) := by
  intro xs domain bounds bounded initial binding
  exact ⟨schedule_safe rest ops encode observe c hc xs bounds bounded _ (fun _ h=>h) initial
      (binding_metadata rest ops encode observe c hc xs domain initial binding),
    correct rest ops encode observe c hc f lower xs domain initial binding⟩

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
