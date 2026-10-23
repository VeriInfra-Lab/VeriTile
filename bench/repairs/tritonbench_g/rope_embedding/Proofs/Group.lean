import bench.repairs.tritonbench_g.rope_embedding.Proofs.Source
namespace VeriTile.Bench.Repairs.RopeEmbedding.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
def groupResult {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) : State α :=
  heads M p (s.pids 0) (coefficient M p s p.cosine p.cosineStride)
    (fun i => if backward then M.unary (some .fp32) (.neg .real) (coefficient M p s p.sine p.sineStride i)
      else coefficient M p s p.sine p.sineStride i)
    (s.pids 1*4) (min (s.pids 1*4+4) p.heads-s.pids 1*4) s

theorem complete_execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) (group : s.pids 1*4≤p.heads) :
    ∃ final, FP.Structural.exec M (source p backward) s=some final ∧
      final.mem=(groupResult M p backward s).mem := by
  obtain ⟨hc,hm,hs,he⟩ := prepared_facts M p backward s
  have first : s.pids 1*4≤min (s.pids 1*4+4) p.heads := le_min (by omega) group
  have stop : s.pids 1*4+(min (s.pids 1*4+4) p.heads-s.pids 1*4)=min (s.pids 1*4+4) p.heads :=
    Nat.add_sub_of_le first
  obtain ⟨final,hloop,_,hmem⟩ := loop_executes M p backward (s.pids 0) _ _
    (s.pids 1*4) (min (s.pids 1*4+4) p.heads-s.pids 1*4) (prepared M p backward s).val hc
  refine ⟨final,?_,?_⟩
  · change run M (source p backward).surfaceBody s=some final
    rw [decomposition,run_append,(prepared M p backward s).property]
    simp only [Option.bind_some,run,step,evalOp_unfold,hs,he,Option.bind]
    simp only [bind,Option.bind]
    rw [← stop,hloop]
  · exact hmem.trans (heads_mem_congr M p (s.pids 0) _ _ _ _ _ s hm)

#axiomsClean complete_execution
end
end VeriTile.Bench.Repairs.RopeEmbedding.Source
