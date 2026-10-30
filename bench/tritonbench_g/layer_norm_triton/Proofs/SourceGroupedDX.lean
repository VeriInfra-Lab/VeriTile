import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedResult
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceRows

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedDX
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph SourceGroupedExecution SourceGroupedResult
noncomputable section

theorem dx_after_store [Inhabited α] (m : Model α) (s : State α) (stride N B cur r j : Nat) :
    dx m (storeDX m s stride N B cur) stride N B r j=dx m s stride N B r j := by
  have hx (o) := storeDX_frame m s stride N B cur "X" o (Or.inl (by decide))
  have hdy (o) := storeDX_frame m s stride N B cur "DY" o (Or.inl (by decide))
  have hw (o) := storeDX_frame m s stride N B cur "W" o (Or.inl (by decide))
  have hm (o) := storeDX_frame m s stride N B cur "Mean" o (Or.inl (by decide))
  have hr (o) := storeDX_frame m s stride N B cur "Rstd" o (Or.inl (by decide))
  have hu : (storeDX m s stride N B cur).undef=s.undef := by
    unfold storeDX
    rw [masked_undef]
  simp only [dx,first,second,xhat,wdy,SourceGroupedGraph.inverse,loaded,hx,hdy,hw,hm,hr,
    hu]

theorem offset_ne (N row cur j k : Nat) (hr : row<cur) (hj : j<N) : row*N+j≠cur*N+k := by
  have hm := Nat.mul_le_mul_right N hr
  nlinarith

theorem fold_preserves_earlier [Inhabited α] (m : Model α) (M N G B group cur row j : Nat)
    (s : State α) (a : Carries α B) (hr : row<cur) (hj : j<N) :
    (fold m N M N G B group cur s a).1.mem "DX" (row*N+j)=s.mem "DX" (row*N+j) := by
  rw [fold]
  split
  · rfl
  · split
    · rw [fold_preserves_earlier m M N G B group (cur+G) row j _ _ (by omega) hj]
      exact storeDX_frame m s N N B cur "DX" (row*N+j)
        (Or.inr (fun k _ => offset_ne N row cur j k hr hj))
    · rfl
termination_by M-cur
decreasing_by omega

theorem fold_dx [Inhabited α] (m : Model α) (M N G B group cur row j : Nat)
    (s : State α) (a : Carries α B) (hg : 0<G) (hb : N≤B) (hj : j<N)
    (hrow : row∈SourceRows.rows M G cur) :
    (fold m N M N G B group cur s a).1.mem "DX" (row*N+j)=.mk .real (dx m s N N B row j) := by
  rw [SourceRows.rows] at hrow
  simp only [Nat.ne_of_gt hg,if_false] at hrow
  split at hrow
  · rename_i hc
    rw [fold]
    simp only [Nat.ne_of_gt hg,if_false,hc,if_true]
    rcases List.mem_cons.mp hrow with he | ht
    · subst row
      rw [fold_preserves_earlier m M N G B group (cur+G) cur j _ _ (by omega) hj]
      exact storeDX_read m s N N B cur j hj hb
    · rw [fold_dx m M N G B group (cur+G) row j _ _ hg hb hj ht,dx_after_store]
  · simp at hrow
termination_by M-cur
decreasing_by omega

theorem final_dx [Inhabited α] (m : Model α) (M N G B row j : Nat)
    (s : State α) (hg : 0<G) (hb : N≤B) (hj : j<N)
    (hrow : row∈groupRows M G (s.pids 0)) :
    (SourceGroupedExecution.final m N M N G B s).mem "DX" (row*N+j)=.mk .real (dx m s N N B row j) := by
  unfold SourceGroupedExecution.final
  rw [stores_frame _ N B (s.pids 0) _ "DX" (row*N+j) (Or.inl ⟨by decide,by decide⟩)]
  exact fold_dx m M N G B (s.pids 0) (s.pids 0) row j s (zero m B) hg hb hj
    (by simpa [SourceRows.rows_eq M G (s.pids 0) hg] using hrow)

#axiomsClean final_dx
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedDX
