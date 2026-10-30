import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelInvariant

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelFrame
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]

def scatterOutside (p : BackwardParameters) (row : Nat) (out : RegionName)
    (stride : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠out ∨ ∀ i : TileIndex [p.block],i.1.val<p.columns → o≠row*stride+i.1.val

def rowOutside (p : BackwardParameters) (row : Nat) (r : RegionName) (o : Nat) : Prop :=
  scatterOutside p row p.dx p.dxStride r o ∧
  (p.recompute=Bool.true → scatterOutside p row p.y p.yStride r o) ∧
  (p.storeDres=Bool.true → scatterOutside p row p.dresIn p.dresInStride r o)

def Outside (p : BackwardParameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (∀ row,Backward.start p s≤row → row<Backward.stop p s → rowOutside p row r o) ∧
  scatterOutside p (s.pids 0) p.dw p.columns r o ∧
  (p.hasBias=Bool.true → scatterOutside p (s.pids 0) p.db p.columns r o)

theorem update (A : Algebra α) (p : BackwardParameters) (row : Nat)
    (a : Backward.Acc α p) (r : RegionName) (o : Nat) (h : rowOutside p row r o) :
    (Backward.update A p row a).state.mem r o=a.state.mem r o := by
  have hy : (Backward.recomputed A p a row).mem r o=a.state.mem r o := by
    cases he : p.recompute
    · simp [Backward.recomputed,he]
    · simpa only [Backward.recomputed,he,if_true] using
        Backward.scatter_frame p row p.y p.yStride _ a.state r o (h.2.1 he)
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ h.1]
  cases he : p.storeDres
  · simpa only [he,Bool.false_eq_true,if_false] using hy
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (h.2.2 he)]
    exact hy

theorem preserves_bounded {β : Type} (step : Nat → β → β) (P : β → Prop)
    (cur stop : Nat) (hstep : ∀ row,cur≤row → row<stop → ∀ a,P a → P (step row a))
    (a : β) (ha : P a) : P (foldRange step cur stop 1 a) := by
  by_cases hc : cur<stop
  · rw [foldRange]
    simp only [Nat.one_ne_zero,hc,if_false,if_true,dite_false,dite_true]
    exact preserves_bounded step P (cur+1) stop
      (fun row hlo hhi b hb => hstep row (by omega) hhi b hb)
      (step cur a) (hstep cur (by omega) hc a ha)
  · simpa [foldRange,hc] using ha
termination_by stop-cur
decreasing_by omega

theorem finish (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (a : Backward.Acc α p) (r : RegionName) (o : Nat)
    (hw : scatterOutside p pid p.dw p.columns r o)
    (hb : p.hasBias=Bool.true → scatterOutside p pid p.db p.columns r o) :
    (Backward.finish A p pid a).mem r o=a.state.mem r o := by
  unfold Backward.finish
  cases he : p.hasBias
  · simpa only [he,Bool.false_eq_true,if_false] using
      Backward.scatter_frame p pid p.dw p.columns _ a.state r o hw
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (hb he),
      Backward.scatter_frame _ _ _ _ _ _ _ _ hw]

theorem reference (A : Algebra α) (p : BackwardParameters) (s : State α)
    (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (Backward.reference A p s).mem r o=s.mem r o := by
  rw [Backward.reference,finish A p (s.pids 0) _ r o h.2.1 h.2.2]
  apply preserves_bounded (Backward.update A p) (fun a => a.state.mem r o=s.mem r o)
  · intro row hlo hhi a ha
    rw [update A p row a r o (h.1 row hlo hhi)]
    exact ha
  · rfl

theorem offset_ne (stride a b row other : Nat) (ha : a<stride) (hb : b<stride)
    (hne : other≠row) : row*stride+a≠other*stride+b := by
  intro he
  rcases lt_trichotomy row other with hl | heq | hr
  · have hm := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt hl)
    nlinarith
  · exact hne heq.symm
  · have hm := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt hr)
    nlinarith

#axiomsClean reference
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelFrame
