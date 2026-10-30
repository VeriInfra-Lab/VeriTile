import bench.tritonbench_g.layer_norm_triton.Proofs.Geometry

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceRows

def rows (M G cur : Nat) : List Nat :=
  if G=0 then [] else if cur<M then cur::rows M G (cur+G) else []
termination_by M-cur
decreasing_by omega

theorem count_step (M G cur : Nat) (hg : 0<G) (hc : cur<M) :
    (M-cur+G-1)/G=(M-(cur+G)+G-1)/G+1 := by
  by_cases h : cur+G≤M
  · have he : M-cur+G-1=(M-(cur+G)+G-1)+G := by omega
    rw [he,Nat.add_div_right _ hg]
  · have hnext : (M-(cur+G)+G-1)/G=0 := by
      apply Nat.div_eq_of_lt
      omega
    rw [hnext]
    have hlow := (Geometry.ceil_mem (M-cur) G 0 hg).mpr (by omega)
    have hhigh : ¬1<(M-cur+G-1)/G := by
      rw [Geometry.ceil_mem (M-cur) G 1 hg]
      omega
    omega

theorem rows_eq (M G cur : Nat) (hg : 0<G) : rows M G cur=groupRows M G cur := by
  rw [rows]
  simp only [Nat.ne_of_gt hg,if_false]
  split
  · rw [rows_eq M G (cur+G) hg]
    unfold groupRows
    rw [count_step M G cur hg ‹cur<M›,List.range'_succ]
  · have hn : (M-cur+G-1)/G=0 := by
      apply Nat.div_eq_of_lt
      omega
    simp [groupRows,hn]
termination_by M-cur
decreasing_by omega

#axiomsClean rows_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormTriton.SourceRows
