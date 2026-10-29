import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Grid

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source.FiniteReadback
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

/-- Equality only at finite source input coordinates. No condition is imposed
on padding, output storage, unrelated regions, registers or unreachable reads. -/
structure Agrees (p : Parameters) (s t : State α) : Prop where
  pids : s.pids=t.pids
  query : ∀ (i j : Fin 128), s.pids 0*128+i.val<p.length →
    (s.mem p.Q (base p s p.q+(s.pids 0*128+i.val)*p.q.row+j.val*p.q.dim)).read .real=
    (t.mem p.Q (base p t p.q+(t.pids 0*128+i.val)*p.q.row+j.val*p.q.dim)).read .real
  queryScale :
    (s.mem p.QS (s.pids 1*((p.length+127)/128)+s.pids 0)).read .real=
    (t.mem p.QS (t.pids 1*((p.length+127)/128)+t.pids 0)).read .real
  key : ∀ row, row<p.length → ∀ (j : Fin 128),
    (s.mem p.K (base p s p.k+j.val*p.k.dim+row*p.k.row)).read .real=
    (t.mem p.K (base p t p.k+j.val*p.k.dim+row*p.k.row)).read .real
  value : ∀ row, row<p.length → ∀ (j : Fin 128),
    (s.mem p.V (base p s p.v+row*p.v.row+j.val*p.v.dim)).read .real=
    (t.mem p.V (base p t p.v+row*p.v.row+j.val*p.v.dim)).read .real
  keyScale : ∀ row, row<p.length →
    (s.mem p.KS (s.pids 1*((p.length+63)/64)+row/64)).read .real=
    (t.mem p.KS (t.pids 1*((p.length+63)/64)+row/64)).read .real

theorem query_eq (M : Algebra α) (p : Parameters) (s t : State α) (same : Agrees p s t) :
    query M p s=query M p t := by
  funext i
  by_cases active : s.pids 0*128+i.1.val<p.length
  · simpa only [query,←same.pids,active,i.2.1.isLt,and_self,ite_true] using
      congrArg M.fp32Load (same.query i.1 i.2.1 active)
  · simp only [query,←same.pids,active,false_and,ite_false]

theorem key_eq (M : Algebra α) (p : Parameters) (s t : State α) (same : Agrees p s t)
    (row : Nat) : key M p s row=key M p t row := by
  funext i
  by_cases active : i.2.1.val<p.length-row
  · simp only [key,active,i.1.isLt,and_self,ite_true]
    rw [same.key (row+i.2.1.val) (by omega)]
  · simp only [key,active,false_and,ite_false]

theorem value_eq (M : Algebra α) (p : Parameters) (s t : State α) (same : Agrees p s t)
    (row : Nat) : value M p s row=value M p t row := by
  funext i
  by_cases active : i.1.val<p.length-row
  · simp only [value,active,i.2.1.isLt,and_self,ite_true]
    rw [same.value (row+i.1.val) (by omega)]
  · simp only [value,active,false_and,ite_false]

theorem update_eq (M : Algebra α) (dot : Dot α) (p : Parameters) (s t : State α)
    (same : Agrees p s t) (row : Nat) (hr : row<p.length) (a : Acc α) :
    update M dot p s row a=update M dot p t row a := by
  have qs : queryScale M p s=queryScale M p t := congrArg M.fp32Load same.queryScale
  have ks : keyScale M p s row=keyScale M p t row := congrArg M.fp32Load (same.keyScale row hr)
  delta update Source.maximum rescale probability scores contribution
  rw [query_eq M p s t same,key_eq M p s t same row,value_eq M p s t same row,qs,ks,same.pids]

theorem fold_eq (M : Algebra α) (dot : Dot α) (p : Parameters) (s t : State α)
    (same : Agrees p s t) (cur finish : Nat) (hf : finish≤p.length) (a : Acc α) :
    foldRange (update M dot p s) cur finish 64 a=
      foldRange (update M dot p t) cur finish 64 a := by
  by_cases active : cur<finish
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [show (64:Nat)≠0 by omega,active,if_false,if_true]
    rw [update_eq M dot p s t same cur (by omega)]
    exact fold_eq M dot p s t same (cur+64) finish hf _
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp [active]
termination_by finish-cur
decreasing_by omega

/-- The complete two-stage attention answer depends only on finite source
reads. This connects old total-memory reference lemmas to finite public IO. -/
theorem answer_eq (M : Algebra α) (dot : Dot α) (p : Parameters) (s t : State α)
    (same : Agrees p s t) (legal : s.pids 0*128<p.length) :
    Whole.answer M dot p s=Whole.answer M dot p t := by
  have first : prefixAcc M dot p s=prefixAcc M dot p t := by
    unfold prefixAcc
    rw [←same.pids]
    exact fold_eq M dot p s t same 0 (s.pids 0*128) (by omega) _
  have last : stop p s=stop p t := by simp only [stop,same.pids]
  unfold Whole.answer accumulated
  rw [←same.pids,←last,←first,
    fold_eq M dot p s t same (s.pids 0*128) (stop p s) (Nat.min_le_right _ _) _]

#axiomsClean answer_eq
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source.FiniteReadback
