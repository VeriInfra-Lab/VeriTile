import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeScheduleFrame

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow TypedSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

theorem raw_reads {α : Type} [Inhabited α] (f : Flags) (r : Regions) (s t : State α)
    (h : ReadsEqual f r s t) (region : RegionName) (hr : Domain.readRegion f r region)
    (base N col : Nat) : raw s region base N col=raw t region base N col := by
  simp only [raw,h region hr]

theorem stream_reads {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (s t : State α)
    (h : ReadsEqual f r s t) (region : RegionName) (hr : Domain.readRegion f r region)
    (base index N : Nat) (p : α) (block : Nat) :
    (stream a f r region base index N p block s).value=(stream a f r region base index N p block t).value := by
  have hs : f.rowscale=Bool.true → s.mem r.rowscale index=t.mem r.rowscale index := by
    intro hf; exact h _ (by simp [Domain.readRegion,hf]) _
  have he : f.dropout=Bool.true → s.mem r.seeds index=t.mem r.seeds index := by
    intro hf; exact h _ (by simp [Domain.readRegion,hf]) _
  funext j
  cases hscale : f.rowscale <;> cases hdrop : f.dropout
  all_goals simp [stream,scalar,raw_reads f r s t h region hr,hscale,hdrop,hs,he]

theorem stream_target {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) (second : Bool) (j : Fin block) :
    (stream a f r (if second then r.x1 else r.x)
      (row*(if second then stride.x1 else stride.x)) (if second then M+row else row) N p block s).value j=
      NativeParameters.stream f (snapshot r stride s) M N p second row j.val := by
  cases second <;> rfl

/-- Pending input graph produced by sequential actual loads and both mask
stores. The equality follows from the read/write frame, not from FP arithmetic
laws. The residual FMA consumes the retained raw/scale register operands. -/
theorem inputStage_value {α : Type} [Inhabited α] (a : Ops α) (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (p : α) (block : Nat) (s : State α) (hsep : Domain.ReadWriteSeparated f r)
    (j : Fin block) :
    (inputStage a f r stride M N row p block s).value j=
      NativeParameters.combined f (snapshot r stride s) M N p row j.val := by
  let first := stream a f r r.x (row*stride.x) row N p block s
  let t := storeMask f r row N block first s
  let second := stream a f r r.x1 (row*stride.x1) (M+row) N p block t
  let u := if f.x1 then storeMask f r (M+row) N block second t else t
  have ht : ReadsEqual f r t s := storeMask_reads f r row N block first s hsep
  have hu : ReadsEqual f r u s := by
    intro region hr offset
    dsimp only [u]
    cases hx : f.x1
    · exact ht region hr offset
    · exact (storeMask_reads f r (M+row) N block second t hsep region hr offset).trans (ht region hr offset)
  have hfirst : first.value j=NativeParameters.stream f (snapshot r stride s) M N p Bool.false row j.val :=
    stream_target a f r stride M N row p block s Bool.false j
  have hsecond : f.x1=Bool.true →
      second.value j=NativeParameters.stream f (snapshot r stride s) M N p Bool.true row j.val := by
    intro hx
    have he := congrFun (stream_reads a f r t s ht r.x1 (by simp [Domain.readRegion,hx])
      (row*stride.x1) (M+row) N p block) j
    exact he.trans (stream_target a f r stride M N row p block s Bool.true j)
  have hres : f.residual=Bool.true → raw u r.residual (row*stride.residual) N j.val=
      NativeRow.load (snapshot r stride s).residual N row j.val := by
    intro hr
    exact raw_reads f r u s hu r.residual (by simp [Domain.readRegion,hr]) _ _ _
  have hraw : first.raw j=NativeRow.load (snapshot r stride s).x N row j.val := rfl
  have hscale : first.scale=Expr.cast32 (.input ((snapshot r stride s).rowscale row)) := rfl
  change (if f.residual then
    if f.rowscale && !f.dropout && !f.x1 then Expr.fma (first.raw j) first.scale
      (raw u r.residual (row*stride.residual) N j.val)
    else Expr.add (if f.x1 then .add (first.value j) (second.value j) else first.value j)
      (raw u r.residual (row*stride.residual) N j.val)
    else if f.x1 then .add (first.value j) (second.value j) else first.value j)=_
  cases hx : f.x1 <;> cases hr : f.residual
  all_goals simp [NativeParameters.combined,hx,hr,hfirst,hsecond,hres,hraw,hscale]

#axiomsClean stream_reads
#axiomsClean stream_target
#axiomsClean inputStage_value
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
