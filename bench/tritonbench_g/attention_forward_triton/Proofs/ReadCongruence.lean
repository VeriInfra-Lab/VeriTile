import bench.tritonbench_g.attention_forward_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Logical
open VeriTile Triton FP.Structural Source Whole
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Equality only at actual source reads, retaining arbitrary memory elsewhere. -/
structure Reads (p : Parameters) (s t : State α) : Prop where
  pids : s.pids=t.pids
  query : ∀ i : TileIndex [128,128], active p s i →
    (s.mem p.Q (base p s p.q+(s.pids 0*128+i.1.val)*p.q.row+i.2.1.val*p.q.dim)).read .real =
    (t.mem p.Q (base p s p.q+(s.pids 0*128+i.1.val)*p.q.row+i.2.1.val*p.q.dim)).read .real
  queryScale :
    (s.mem p.QS (s.pids 1*((p.length+127)/128)+s.pids 0)).read .real =
    (t.mem p.QS (s.pids 1*((p.length+127)/128)+s.pids 0)).read .real
  key : ∀ row, row<p.length → ∀ i : TileIndex [128,64],
    i.2.1.val<p.length-row ∧ i.1.val<96 →
    (s.mem p.K (base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row)).read .real =
    (t.mem p.K (base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row)).read .real
  keyScale : ∀ row, row<p.length →
    (s.mem p.KS (s.pids 1*((p.length+63)/64)+row/64)).read .real =
    (t.mem p.KS (s.pids 1*((p.length+63)/64)+row/64)).read .real
  value : ∀ row, row<p.length → ∀ i : TileIndex [64,128],
    i.1.val<p.length-row ∧ i.2.1.val<96 →
    (s.mem p.V (base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim)).read .real =
    (t.mem p.V (base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim)).read .real

theorem answer_eq (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s t : State α) (reads : Reads p s t) : answer M dot p s=answer M dot p t := by
  have baseEq (st : Strides) : base p t st=base p s st := by simp only [base,←reads.pids]
  have queryEq : query M p s=query M p t := by
    funext i
    simp only [query,baseEq,←reads.pids]
    split_ifs with active
    · rw [reads.query i active]
    · rfl
  have scaleEq : queryScale M p s=queryScale M p t := by
    simp only [queryScale,←reads.pids,reads.queryScale]
  have stepEq (row : Nat) (hr : row<p.length) : update M dot p s row=update M dot p t row := by
    have keyEq : key M p s row=key M p t row := by
      funext i
      simp only [key,baseEq]
      split_ifs with active
      · rw [reads.key row hr i active]
      · rfl
    have valueEq : value M p s row=value M p t row := by
      funext i
      simp only [value,baseEq]
      split_ifs with active
      · rw [reads.value row hr i active]
      · rfl
    have keyScaleEq : keyScale M p s row=keyScale M p t row := by
      simp only [keyScale,←reads.pids,reads.keyScale row hr]
    funext a
    delta update maximum rescale probability scores contribution
    simp only [queryEq,scaleEq,keyEq,valueEq,keyScaleEq]
  let rec foldEq (cur : Nat) (a : Acc α) :
      foldRange (update M dot p s) cur p.length 64 a=
        foldRange (update M dot p t) cur p.length 64 a := by
    by_cases active : cur<p.length
    · conv_lhs => rw [foldRange]
      conv_rhs => rw [foldRange]
      simp only [show ¬(64:Nat)=0 by omega,active,if_false,if_true]
      rw [stepEq cur active]
      exact foldEq (cur+64) _
    · simp [foldRange,active]
    termination_by p.length-cur
    decreasing_by omega
  simp only [answer,accumulated,foldEq]

#axiomsClean answer_eq
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Logical
