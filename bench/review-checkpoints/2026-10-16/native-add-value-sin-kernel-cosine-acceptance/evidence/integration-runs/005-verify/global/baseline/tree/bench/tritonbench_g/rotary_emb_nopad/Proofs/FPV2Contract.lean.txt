import bench.tritonbench_g.rotary_emb_nopad.Proofs.FPV2
import bench.tritonbench_g.rotary_emb_nopad.Proofs.Pair
namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPV2
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def Post {α : Type} [Inhabited α] (cache : Bool) (p : Parameters) (s final : State α)
    (results : Bool → TileIndex [p.half] → α) : Prop :=
  (∀ second i, active p s i → final.mem (outputRegion cache p) (outputOffset cache p s second i) = .mk .fp32 (results second i)) ∧
  (∀ r o, Pair.Outside (outputRegion cache p) (active p s) (outputOffset cache p s) r o → final.mem r o = s.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (cache : Bool)
    (p : Parameters) (s : State α) (hw : Pair.WritesApart (active p s) (outputOffset cache p s)) :
    ∃ final, FP.Structural.exec M (source cache p) s = some final ∧
      Post cache p s final (value M p s) := by
  refine ⟨(execution M cache p s).val, (execution M cache p s).property, ?_, ?_⟩
  · intro second i hi
    rw [execution_memory]
    exact Pair.readback (outputRegion cache p) (active p s) (outputOffset cache p s)
      (fun second i => .mk .fp32 (value M p s second i)) s hw second i hi
  · intro r o h
    rw [execution_memory]
    exact Pair.frame (outputRegion cache p) (active p s) (outputOffset cache p s)
      (fun second i => .mk .fp32 (value M p s second i)) s r o h

#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPV2
