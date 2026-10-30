import bench.tritonbench_g.rotary_emb_nopad.Proofs.FPBlocks
import bench.tritonbench_g.rotary_emb_nopad.Proofs.Pair
namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPBlocks
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def Post {α : Type} [Inhabited α] (key : Bool) (p : Parameters) (s final : State α)
    (results : Bool → Index p → α) : Prop :=
  (∀ second i, active key p s i → final.mem (p.input) (offset key p s second i) = .mk .fp32 (results second i)) ∧
  (∀ r o, Pair.Outside (p.input) (active key p s) (offset key p s) r o → final.mem r o = s.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (key : Bool)
    (p : Parameters) (s : State α) (hw : Pair.WritesApart (active key p s) (offset key p s)) :
    ∃ final, FP.Structural.exec M (source key p) s = some final ∧
      Post key p s final (value M key p s) := by
  refine ⟨(execution M key p s).val, (execution M key p s).property, ?_, ?_⟩
  · intro second i hi
    rw [execution_memory]
    exact Pair.readback (p.input) (active key p s) (offset key p s)
      (fun second i => .mk .fp32 (value M key p s second i)) s hw second i hi
  · intro r o h
    rw [execution_memory]
    exact Pair.frame (p.input) (active key p s) (offset key p s)
      (fun second i => .mk .fp32 (value M key p s second i)) s r o h

#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPBlocks
