import bench.tritonbench_g.layer_norm_liger.Proofs.LogicalForward
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
noncomputable section

theorem forward_all_exec (rest : Algebra ℝ) (p : Forward.Parameters) (s : State ℝ)
    (hb : p.columns≤p.block) (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hry : p.rstd≠p.y) :
    ∃ t, FP.Structural.exec (withSum (algebra rest) sumPrimitive) (Forward.kernel p) s=some t ∧
      t.mem p.mean (Forward.meanOffset p s)=.mk .fp32 (mean p.columns (rowX p s)) ∧
      t.mem p.rstd (Forward.rstdOffset p s)=.mk .fp32 (rstd p.columns p.epsilon (rowX p s)) ∧
      (∀ i : TileIndex [p.block], i.1.val<p.columns →
        t.mem p.y (Forward.outputOffset p s i)=.mk .fp32
          (y p.columns p.epsilon (rowX p s) (weight p s) (bias p s) i.1.val)) ∧
      (∀ r o, Forward.Outside p s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,ht,hm⟩ := Forward.complete_run (algebra rest) sumPrimitive p s
  refine ⟨t,ht,?_,?_,?_,?_⟩
  · rw [hm,Forward.reference_mean _ _ _ _ hmy hmr,forward_mean rest p s hb]
    rfl
  · rw [hm,Forward.reference_rstd _ _ _ _ hry,forward_rstd rest p s hb]
    rfl
  · intro i hi
    rw [hm,Forward.reference_y _ _ _ _ i hi,forward_y rest p s hb i hi]
    rfl
  · intro r o ho
    rw [hm]
    exact Forward.reference_frame _ _ _ _ r o ho

#axiomsClean forward_all_exec
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
