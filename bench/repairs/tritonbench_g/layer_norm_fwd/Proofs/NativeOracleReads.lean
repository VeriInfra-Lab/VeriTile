import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeScheduleReturn

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow TypedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

theorem stream_snapshot_reads {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions)
    (stride : Strides) (M N row col : Nat) (p : α) (s t : State α) (h : ReadsEqual f r s t)
    (second : Bool) (hx : second=Bool.true → f.x1=Bool.true) :
    NativeParameters.stream f (snapshot r stride s) M N p second row col=
      NativeParameters.stream f (snapshot r stride t) M N p second row col := by
  have hr : Domain.readRegion f r (if second then r.x1 else r.x) := by
    cases second <;> simp_all [Domain.readRegion]
  have he := congrFun (stream_reads a f r s t h (if second then r.x1 else r.x) hr
    (row*(if second then stride.x1 else stride.x)) (if second then M+row else row) N p (col+1)) ⟨col,by omega⟩
  rw [stream_target,stream_target] at he
  exact he

theorem combined_snapshot_reads {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions)
    (stride : Strides) (M N row col : Nat) (p : α) (s t : State α) (h : ReadsEqual f r s t) :
    NativeParameters.combined f (snapshot r stride s) M N p row col=
      NativeParameters.combined f (snapshot r stride t) M N p row col := by
  have hfirst := stream_snapshot_reads a f r stride M N row col p s t h Bool.false (by simp)
  have hsecond := fun (hx : f.x1=Bool.true) => stream_snapshot_reads a f r stride M N row col p s t h Bool.true (fun _ => hx)
  have hraw : NativeRow.load (snapshot r stride s).x N row col=NativeRow.load (snapshot r stride t).x N row col :=
    raw_reads f r s t h r.x (by simp [Domain.readRegion]) (row*stride.x) N col
  have hres : f.residual=Bool.true →
      NativeRow.load (snapshot r stride s).residual N row col=NativeRow.load (snapshot r stride t).residual N row col := by
    intro hr
    exact raw_reads f r s t h r.residual (by simp [Domain.readRegion,hr]) (row*stride.residual) N col
  have hscale : f.rowscale=Bool.true → (snapshot r stride s).rowscale row=(snapshot r stride t).rowscale row := by
    intro hs
    exact congrArg (fun v : Cell α => v.read .real) (h r.rowscale (by simp [Domain.readRegion,hs]) row)
  cases hx : f.x1 <;> cases hr : f.residual <;> cases hs : f.rowscale <;> cases hd : f.dropout
  all_goals simp [NativeParameters.combined,hx,hr,hs,hd,hfirst,hsecond,hraw,hres,hscale]

/-- The independent complete target depends only on enabled input cells.
Disabled optional input handles may alias outputs and need not be preserved. -/
theorem oracle_reads {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (s t : State α) (h : ReadsEqual f r s t) :
    oracle a red plan plan1 f r stride M N eps p s=oracle a red plan plan1 f r stride M N eps p t := by
  have hc := fun row col => combined_snapshot_reads a f r stride M N row col p s t h
  have hm : ∀ row col, NativeParameters.mean red f (snapshot r stride s) M N p row col=
      NativeParameters.mean red f (snapshot r stride t) M N p row col := by
    intro row col
    simp only [NativeParameters.mean,hc]
  have hcenter : ∀ row col, NativeParameters.centered red f (snapshot r stride s) M N p row col=
      NativeParameters.centered red f (snapshot r stride t) M N p row col := by
    intro row col
    simp only [NativeParameters.centered,hc,hm]
  have hv : ∀ row col, NativeParameters.variance red f (snapshot r stride s) M N p row col=
      NativeParameters.variance red f (snapshot r stride t) M N p row col := by
    intro row col
    simp only [NativeParameters.variance,hcenter]
  have hs : ∀ row col, NativeParameters.rstd red f (snapshot r stride s) M N eps p row col=
      NativeParameters.rstd red f (snapshot r stride t) M N eps p row col := by
    intro row col
    simp only [NativeParameters.rstd,hcenter,hv]
  have hn : ∀ row col, NativeParameters.normalized red f (snapshot r stride s) M N eps p row col=
      NativeParameters.normalized red f (snapshot r stride t) M N eps p row col := by
    intro row col
    simp only [NativeParameters.normalized,hcenter,hs]
  have hw : ∀ col, (snapshot r stride s).weight col=(snapshot r stride t).weight col := by
    intro col
    exact congrArg (fun v : Cell α => v.read .real) (h r.weight (by simp [Domain.readRegion]) col)
  have hb : f.bias=Bool.true → ∀ col, (snapshot r stride s).bias col=(snapshot r stride t).bias col := by
    intro hf col
    exact congrArg (fun v : Cell α => v.read .real) (h r.bias (by simp [Domain.readRegion,hf]) col)
  have hw1 : f.w1=Bool.true → ∀ col, (snapshot r stride s).weight1 col=(snapshot r stride t).weight1 col := by
    intro hf col
    exact congrArg (fun v : Cell α => v.read .real) (h r.weight1 (by simp [Domain.readRegion,hf]) col)
  have hb1 : f.w1=Bool.true → f.b1=Bool.true → ∀ col,
      (snapshot r stride s).bias1 col=(snapshot r stride t).bias1 col := by
    intro hf hbf col
    exact congrArg (fun v : Cell α => v.read .real) (h r.bias1 (by simp [Domain.readRegion,hf,hbf]) col)
  have hy : ∀ row col, NativeParameters.y red plan f (snapshot r stride s) M N eps p row col=
      NativeParameters.y red plan f (snapshot r stride t) M N eps p row col := by
    intro row col
    cases hf : f.bias <;> simp only [NativeParameters.y,NativeRow.affine,hn,hw,hf,Bool.false_eq_true,ite_false,ite_true]
    all_goals first | rfl | rw [hb hf]
  have hy1 : f.w1=Bool.true → ∀ row col, NativeParameters.y1 red plan1 f (snapshot r stride s) M N eps p row col=
      NativeParameters.y1 red plan1 f (snapshot r stride t) M N eps p row col := by
    intro hf row col
    cases hbf : f.b1 <;> simp only [NativeParameters.y1,NativeRow.affine,hn,hw1 hf,hbf,Bool.false_eq_true,ite_false,ite_true]
    all_goals first | rfl | rw [hb1 hf hbf]
  have hx := h r.x (by simp [Domain.readRegion])
  have hseed : f.dropout=Bool.true → ∀ i, s.mem r.seeds i=t.mem r.seeds i := by
    intro hd
    exact h r.seeds (by simp [Domain.readRegion,hd])
  apply TypedSource.Result.ext
  all_goals dsimp only [oracle]
  · simp only [hy]
  · cases hf : f.w1 <;> simp [hf,hy1]
  · simp only [hm]
  · simp only [hs]
  · simp only [hc,hx]
  · cases hd : f.dropout <;> simp [hd,hseed]
  · cases hd : f.dropout <;> simp [hd,snapshot,hseed]
  · cases hd : f.dropout <;> simp [hd,snapshot,hseed]

#axiomsClean stream_snapshot_reads
#axiomsClean combined_snapshot_reads
#axiomsClean oracle_reads
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
