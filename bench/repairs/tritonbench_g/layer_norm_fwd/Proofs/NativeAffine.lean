import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeNormalization

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

theorem affine_preserves {α : Type} [Inhabited α] (a : Ops α) (plan plan1 : AffinePlan)
    (f : Flags) (r : Regions) (stride : Strides) (N row : Nat) (input : NormalRegisters α block)
    (region : RegionName) (offset : Nat) (hy : region≠r.y) (hy1 : region≠r.y1) :
    (affineStage a plan plan1 f r stride N row input).mem region offset=input.memory.mem region offset := by
  dsimp only [affineStage]
  cases hf : f.w1
  · rw [if_neg Bool.false_ne_true]
    exact vectorStore_frame _ r.y _ _ _ _ region offset (by simp [hy])
  · rw [if_pos rfl,vectorStore_frame _ r.y1 _ _ _ _ region offset (by simp [hy1])]
    exact vectorStore_frame _ r.y _ _ _ _ region offset (by simp [hy])

theorem normalized_target {α : Type} (red : Nat → Whole.Reductions block)
    (f : Flags) (M N row : Nat) (eps p : α) (x : NativeRow.Inputs α) (input : NormalRegisters α block)
    (hc : ∀ j, input.centered j=NativeParameters.centered red f x M N p row j.val)
    (hr : ∀ owner, input.rstd owner=NativeParameters.rstd red f x M N eps p row owner) (j : Fin block) :
    (if f.rms then Expr.mul (input.rstd j.val) (input.centered j)
      else Expr.mul (input.centered j) (input.rstd j.val))=NativeParameters.normalized red f x M N eps p row j.val := by
  simp only [NativeParameters.normalized,hc,hr]

theorem affine_y_store {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (s : State α) (input : NormalRegisters α block)
    (hread : ReadsEqual f r input.memory s) (hysep : r.y≠r.y1)
    (hc : ∀ j, input.centered j=NativeParameters.centered red f (snapshot r stride s) M N p row j.val)
    (hr : ∀ owner, input.rstd owner=NativeParameters.rstd red f (snapshot r stride s) M N eps p row owner)
    (j : Fin block) (hj : j.val<N) :
    (affineStage a plan plan1 f r stride N row input).mem r.y (row*stride.y+j.val)=
      Cell.mk .real ((NativeParameters.y red plan f (snapshot r stride s) M N eps p row j.val).eval a) := by
  have hw := hread r.weight (by simp [Domain.readRegion]) j.val
  have hb : f.bias=Bool.true → input.memory.mem r.bias j.val=s.mem r.bias j.val := by
    intro hf; exact hread _ (by simp [Domain.readRegion,hf]) _
  have hn := normalized_target red f M N row eps p (snapshot r stride s) input hc hr j
  dsimp only [affineStage]
  cases hf : f.w1
  all_goals try rw [if_neg Bool.false_ne_true]
  all_goals try rw [if_pos rfl,vectorStore_frame _ r.y1 _ _ _ _ r.y (row*stride.y+j.val) (by simp [hysep])]
  all_goals rw [vectorStore_read _ _ _ _ _ _ j hj]
  all_goals rw [hn]
  all_goals cases hbias : f.bias
  all_goals simp [NativeParameters.y,NativeRow.affine,raw,snapshot,hj,hw,hb,hbias]

theorem affine_y1_store {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (s : State α) (input : NormalRegisters α block)
    (hread : ReadsEqual f r input.memory s) (hsep : Domain.ReadWriteSeparated f r)
    (hc : ∀ j, input.centered j=NativeParameters.centered red f (snapshot r stride s) M N p row j.val)
    (hr : ∀ owner, input.rstd owner=NativeParameters.rstd red f (snapshot r stride s) M N eps p row owner)
    (hf : f.w1=Bool.true) (j : Fin block) (hj : j.val<N) :
    (affineStage a plan plan1 f r stride N row input).mem r.y1 (row*stride.y1+j.val)=
      Cell.mk .real ((NativeParameters.y1 red plan1 f (snapshot r stride s) M N eps p row j.val).eval a) := by
  have hwsep : r.weight1≠r.y := hsep r.weight1 (by simp [Domain.readRegion,hf]) r.y (by simp [Domain.writeRegion])
  have hbsep : f.b1=Bool.true → r.bias1≠r.y := by
    intro hb; exact hsep r.bias1 (by simp [Domain.readRegion,hf,hb]) r.y (by simp [Domain.writeRegion])
  have hw := hread r.weight1 (by simp [Domain.readRegion,hf]) j.val
  have hb : f.b1=Bool.true → input.memory.mem r.bias1 j.val=s.mem r.bias1 j.val := by
    intro hb; exact hread _ (by simp [Domain.readRegion,hf,hb]) _
  have hn := normalized_target red f M N row eps p (snapshot r stride s) input hc hr j
  dsimp only [affineStage]
  rw [if_pos hf,vectorStore_read _ _ _ _ _ _ j hj,hn]
  cases hbias : f.b1
  · simp only [NativeParameters.y1,NativeRow.affine,hbias,Bool.false_eq_true,ite_false,raw,if_pos hj,Nat.zero_add]
    rw [vectorStore_frame _ r.y _ _ _ _ r.weight1 j.val (by simp [hwsep]),hw]
    rfl
  · simp only [NativeParameters.y1,NativeRow.affine,hbias,ite_true,raw,if_pos hj,Nat.zero_add]
    rw [vectorStore_frame _ r.y _ _ _ _ r.weight1 j.val (by simp [hwsep]),
      vectorStore_frame _ r.y _ _ _ _ r.bias1 j.val (by simp [hbsep hbias]),hw,hb hbias]
    rfl

#axiomsClean affine_preserves
#axiomsClean normalized_target
#axiomsClean affine_y_store
#axiomsClean affine_y1_store
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
