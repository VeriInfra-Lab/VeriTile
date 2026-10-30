import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalGraph

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceFinalResult
open VeriTile.Triton FP.Structural SourceModel SourceFinalGraph
noncomputable section

theorem columns_injective (s : State α) (BN : Nat) : Function.Injective (columns s BN) := by
  intro a b h
  apply Prod.ext
  · apply Fin.ext
    exact Nat.add_left_cancel h
  · exact Subsingleton.elim _ _

theorem stores_dw (m : Model α) (s : State α) (N BM BN : Nat) (a : Carries α BM BN)
    (i : TileIndex [BN]) (hi : columns s BN i<N) :
    (stores m s N BM BN a).mem "FINAL_DW" (columns s BN i)=
      .mk .real (SourceForwardGraph.cast m (reduced m BM BN a.1 i)) := by
  unfold stores
  rw [State.masked_scatter_frame]
  · exact State.masked_scatter_readback _ _ _ _ _ i hi (columns_injective s BN)
  · left; decide

theorem stores_db (m : Model α) (s : State α) (N BM BN : Nat) (a : Carries α BM BN)
    (i : TileIndex [BN]) (hi : columns s BN i<N) :
    (stores m s N BM BN a).mem "FINAL_DB" (columns s BN i)=
      .mk .real (SourceForwardGraph.cast m (reduced m BM BN a.2 i)) := by
  exact State.masked_scatter_readback _ _ _ (fun k => columns s BN k<N) _ i hi (columns_injective s BN)

def Outside (s : State α) (N BN : Nat) (r : RegionName) (o : Nat) : Prop :=
  (r≠"FINAL_DW" ∧ r≠"FINAL_DB") ∨ ∀ i : TileIndex [BN],columns s BN i<N → o≠columns s BN i

theorem stores_frame (m : Model α) (s : State α) (N BM BN : Nat) (a : Carries α BM BN)
    (r : RegionName) (o : Nat) (ho : Outside s N BN r o) :
    (stores m s N BM BN a).mem r o=s.mem r o := by
  unfold stores
  rw [State.masked_scatter_frame,State.masked_scatter_frame]
  · rcases ho with hr | hi
    · exact Or.inl hr.1
    · exact Or.inr (fun i _ => hi i)
  · rcases ho with hr | hi
    · exact Or.inl hr.2
    · exact Or.inr (fun i _ => hi i)

theorem executes [Inhabited α] (m : Model α) (s : State α) (M N BM BN : Nat) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) s=some t ∧
      t.mem=(final m s M N BM BN).mem ∧
      (∀ i : TileIndex [BN],columns s BN i<N →
        t.mem "FINAL_DW" (columns s BN i)=
          .mk .real (SourceForwardGraph.cast m (reduced m BM BN (fold m s M N BM BN 0 (zero m BM BN)).1 i)) ∧
        t.mem "FINAL_DB" (columns s BN i)=
          .mk .real (SourceForwardGraph.cast m (reduced m BM BN (fold m s M N BM BN 0 (zero m BM BN)).2 i))) ∧
      (∀ r o,Outside s N BN r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,ht,hm⟩ := source_refines m s M N BM BN
  refine ⟨t,ht,hm,?_,?_⟩
  · intro i hi
    rw [hm]
    exact ⟨stores_dw m s N BM BN _ i hi,stores_db m s N BM BN _ i hi⟩
  · intro r o ho
    rw [hm]
    exact stores_frame m s N BM BN _ r o ho

#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceFinalResult
