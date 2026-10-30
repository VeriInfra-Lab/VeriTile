import bench.tritonbench_g.iv_dependent_matmul.Sources.Mathematical_FPContract

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section

/-- Readback needs uniqueness only among lanes that actually store. -/
theorem masked_readback_active {α : Type} {shape : TileShape} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (enabled : TileIndex shape → Prop) [DecidablePred enabled] (s : State α)
    (i : TileIndex shape) (hi : enabled i)
    (hinj : ∀ j, enabled j → off i=off j → i=j) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if enabled k then t.write r (off k) (val k) else t) s).mem r (off i)=val i := by
  rw [State.masked_scatter_eq_filter]
  let l := (TileShape.allIndices shape).filter (fun j => decide (enabled j))
  have hmem : i∈l := by simp [l,TileShape.mem_allIndices,hi]
  have hnd : l.Nodup := (TileShape.allIndices_nodup shape).filter _
  obtain ⟨before,after,hl⟩ := List.append_of_mem hmem
  rw [hl,List.nodup_append,List.nodup_cons] at hnd
  obtain ⟨_,⟨hnot,_⟩,_⟩ := hnd
  change (l.foldl _ s).mem r (off i)=_
  rw [hl,List.foldl_append,List.foldl_cons,State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj he
    have hjl : j∈l := by rw [hl]; simp [hj]
    have hjEnabled : enabled j := by
      have hh := (List.mem_filter.mp hjl).2
      simpa only [decide_eq_true_eq] using hh
    exact hnot ((hinj j hjEnabled he) ▸ hj)

theorem output_injective_active (p : Parameters) (s : State Word)
    (hStride : p.outputColumnStride=1) (hWidth : p.columns≤p.outputRowStride)
    (i j : TileIndex [p.blockRows,p.blockColumns])
    (hi : active p s i=Bool.true) (hj : active p s j=Bool.true)
    (he : outputOffset p s i=outputOffset p s j) : i=j := by
  have hi' : col p s i.2.1<p.outputRowStride := by
    simp only [active,Bool.and_eq_true,decide_eq_true_eq] at hi
    exact lt_of_lt_of_le hi.2 hWidth
  have hj' : col p s j.2.1<p.outputRowStride := by
    simp only [active,Bool.and_eq_true,decide_eq_true_eq] at hj
    exact lt_of_lt_of_le hj.2 hWidth
  have hn : 0<p.outputRowStride := by omega
  simp only [outputOffset,hStride,Nat.one_mul] at he
  have hm := congrArg (fun n => n%p.outputRowStride) he
  simp [Nat.add_mod,Nat.mod_eq_of_lt hi',Nat.mod_eq_of_lt hj'] at hm
  rw [hm] at he
  have hr := Nat.eq_of_mul_eq_mul_left hn (Nat.add_right_cancel he)
  have h1 : i.1=j.1 := Fin.ext (Nat.add_left_cancel hr)
  have h2 : i.2.1=j.2.1 := Fin.ext (Nat.add_left_cancel hm)
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  cases h1
  cases h2
  rfl

theorem written_post_active [Inhabited Word] (cfg : FP.Config) (p : Parameters) (s : State Word)
    (a : Accumulator p)
    (hinj : ∀ i j, active p s i=Bool.true → active p s j=Bool.true → outputOffset p s i=outputOffset p s j → i=j) :
    Post cfg p s (written cfg p s a) a := by
  constructor
  · intro i hi
    exact masked_readback_active _ _ _ (fun j => active p s j=Bool.true) s i hi (hinj i · hi)
  · intro r o ho
    exact State.masked_scatter_frame _ _ _ (fun i => active p s i=Bool.true) _ r o
      (ho.imp_right fun hh i _ => hh i) s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
