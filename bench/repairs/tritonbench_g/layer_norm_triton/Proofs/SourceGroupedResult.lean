import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedResult
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph SourceGroupedExecution
noncomputable section

theorem storeDX_frame [Inhabited α] (m : Model α) (s : State α) (stride N B cur : Nat)
    (r : RegionName) (o : Nat) (h : r≠"DX" ∨ ∀ j,j<N → o≠cur*stride+j) :
    (storeDX m s stride N B cur).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases h with hr | hi
  · exact Or.inl hr
  · exact Or.inr (fun i _ => hi i.1.val)

theorem storeDX_read [Inhabited α] (m : Model α) (s : State α) (stride N B cur : Nat)
    (j : Nat) (hj : j<N) (hb : N≤B) :
    (storeDX m s stride N B cur).mem "DX" (cur*stride+j)=.mk .real (dx m s stride N B cur j) := by
  let i : TileIndex [B] := (⟨j,by omega⟩,PUnit.unit)
  apply State.masked_scatter_readback (active:=fun k : TileIndex [B] => k.1.val<N) (i:=i)
  · exact hj
  · intro a b h
    apply Prod.ext
    · exact Fin.ext (Nat.add_left_cancel h)
    · exact Subsingleton.elim _ _

def OutsideDX (stride M N : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠"DX" ∨ ∀ row,row<M → ∀ j,j<N → o≠row*stride+j

theorem fold_frame [Inhabited α] (m : Model α) (stride M N G B group cur : Nat)
    (s : State α) (a : Carries α B) (r : RegionName) (o : Nat) (h : OutsideDX stride M N r o) :
    (fold m stride M N G B group cur s a).1.mem r o=s.mem r o := by
  rw [fold]
  split
  · rfl
  · split
    · rw [fold_frame m stride M N G B group (cur+G) _ _ r o h]
      apply storeDX_frame
      rcases h with hr | hi
      · exact Or.inl hr
      · exact Or.inr (hi cur ‹cur<M›)
    · rfl
termination_by M-cur
decreasing_by omega

theorem stores_dw (s : State α) (N B group : Nat) (a : Carries α B)
    (i : TileIndex [B]) (hi : i.1.val<N) :
    (stores s N B group a).mem "DW" (group*N+i.1.val)=.mk .real (a.1 i) := by
  unfold stores
  rw [State.masked_scatter_frame]
  · apply State.masked_scatter_readback (active:=fun k : TileIndex [B] => k.1.val<N) (i:=i)
    · exact hi
    · intro a b h
      apply Prod.ext
      · exact Fin.ext (Nat.add_left_cancel h)
      · exact Subsingleton.elim _ _
  · left; decide

theorem stores_db (s : State α) (N B group : Nat) (a : Carries α B)
    (i : TileIndex [B]) (hi : i.1.val<N) :
    (stores s N B group a).mem "DB" (group*N+i.1.val)=.mk .real (a.2 i) := by
  apply State.masked_scatter_readback (active:=fun k : TileIndex [B] => k.1.val<N) (i:=i)
  · exact hi
  · intro a b h
    apply Prod.ext
    · exact Fin.ext (Nat.add_left_cancel h)
    · exact Subsingleton.elim _ _

def Outside (stride M N group : Nat) (r : RegionName) (o : Nat) : Prop :=
  OutsideDX stride M N r o ∧
    ((r≠"DW" ∧ r≠"DB") ∨ ∀ j,j<N → o≠group*N+j)

theorem stores_frame (s : State α) (N B group : Nat) (a : Carries α B)
    (r : RegionName) (o : Nat) (h : (r≠"DW" ∧ r≠"DB") ∨ ∀ j,j<N → o≠group*N+j) :
    (stores s N B group a).mem r o=s.mem r o := by
  unfold stores
  rw [State.masked_scatter_frame,State.masked_scatter_frame]
  · rcases h with hr | hi
    · exact Or.inl hr.1
    · exact Or.inr (fun i _ => hi i.1.val)
  · rcases h with hr | hi
    · exact Or.inl hr.2
    · exact Or.inr (fun i _ => hi i.1.val)

theorem final_frame [Inhabited α] (m : Model α) (stride M N G B : Nat) (s : State α)
    (r : RegionName) (o : Nat) (h : Outside stride M N (s.pids 0) r o) :
    (final m stride M N G B s).mem r o=s.mem r o := by
  unfold final
  rw [stores_frame _ N B (s.pids 0) _ r o h.2]
  exact fold_frame m stride M N G B (s.pids 0) (s.pids 0) s (zero m B) r o h.1

theorem executes [Inhabited α] (m : Model α) (stride M N G B : Nat) (s : State α) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s=some t ∧
      t.mem=(final m stride M N G B s).mem ∧
      (∀ i : TileIndex [B],i.1.val<N →
        t.mem "DW" (s.pids 0*N+i.1.val)=
          .mk .real ((fold m stride M N G B (s.pids 0) (s.pids 0) s (zero m B)).2.1 i) ∧
        t.mem "DB" (s.pids 0*N+i.1.val)=
          .mk .real ((fold m stride M N G B (s.pids 0) (s.pids 0) s (zero m B)).2.2 i)) ∧
      (∀ r o,Outside stride M N (s.pids 0) r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,ht,hm⟩ := source_refines m stride M N G B s
  refine ⟨t,ht,hm,?_,?_⟩
  · intro i hi
    rw [hm]
    exact ⟨stores_dw _ N B (s.pids 0) _ i hi,stores_db _ N B (s.pids 0) _ i hi⟩
  · intro r o ho
    rw [hm]
    exact final_frame m stride M N G B s r o ho

#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedResult
