import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeGeometry
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeAffine
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeInputStores

/-! Complete stored output of the explicit native schedule, with arbitrary
opaque arithmetic. Source/compiler correspondence remains a separate structural
obligation; this theorem does not infer it from an arithmetic exactness law. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

def RowValues {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (s t : State α) : Prop :=
  let x := snapshot r stride s
  (f.residualOut=Bool.true → ∀ j : Fin block, j.val<N →
    t.mem r.residualOut (row*stride.residualOut+j.val)=Cell.mk .real
      (a.cast32 ((NativeParameters.combined f x M N p row j.val).eval a))) ∧
  (f.dropout=Bool.true → f.dropoutMask=Bool.true → ∀ j : Fin block, j.val<N →
    t.mem r.dropoutMask (row*N+j.val)=Cell.mk .bool (keep a (x.seeds row) p j.val)) ∧
  (f.dropout=Bool.true → f.dropoutMask=Bool.true → f.x1=Bool.true → ∀ j : Fin block, j.val<N →
    t.mem r.dropoutMask ((M+row)*N+j.val)=Cell.mk .bool (keep a (x.seeds (M+row)) p j.val)) ∧
  (∀ j : Fin block, j.val<N → t.mem r.y (row*stride.y+j.val)=Cell.mk .real
    ((NativeParameters.y red plan f x M N eps p row j.val).eval a)) ∧
  (f.w1=Bool.true → ∀ j : Fin block, j.val<N → t.mem r.y1 (row*stride.y1+j.val)=Cell.mk .real
    ((NativeParameters.y1 red plan1 f x M N eps p row j.val).eval a)) ∧
  t.mem r.rstd row=Cell.mk .real (a.cast32 ((NativeParameters.rstd red f x M N eps p row 0).eval a)) ∧
  (f.rms=Bool.false → t.mem r.mean row=Cell.mk .real
    (a.cast32 ((NativeParameters.mean red f x M N p row 0).eval a)))

theorem execute_row_post {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (hblock : 0<block) (s : State α)
    (hsep : Domain.ReadWriteSeparated f r) (hgeo : NativeGeometry.Geometry f r stride M N) (hrow : row<M) :
    TypedSource.Frame f r stride M N row s (execute a red plan plan1 f r stride M N row eps p hblock s) ∧
    RowValues a red plan plan1 f r stride M N row eps p s
      (execute a red plan plan1 f r stride M N row eps p hblock s) := by
  have hn := hgeo.outputs
  simp only [List.nodup_cons,List.mem_cons,List.not_mem_nil,not_or,List.nodup_nil,and_true] at hn
  have h01 : r.y≠r.y1 := by simp_all [ne_comm]
  have h12 : r.rstd≠r.y := by simp_all [ne_comm]
  have h02 : r.rstd≠r.y1 := by simp_all [ne_comm]
  have h13 : r.mean≠r.y := by simp_all [ne_comm]
  have h03 : r.mean≠r.y1 := by simp_all [ne_comm]
  have h23 : r.mean≠r.rstd := by simp_all [ne_comm]
  have hmask : r.dropoutMask≠r.residualOut := by simp_all [ne_comm]
  have hkeepnames : ∀ region, region=r.residualOut ∨ region=r.dropoutMask →
      region≠r.y ∧ region≠r.y1 ∧ region≠r.rstd ∧ region≠r.mean := by
    intro region hr
    rcases hr with h|h <;> subst region <;> simp_all [ne_comm]
  let first := inputStage a f r stride M N row p block s
  let normal := normalizationStage a red f r N row eps hblock first
  let final := affineStage a plan plan1 f r stride N row normal
  have hv : ∀ j, first.value j=NativeParameters.combined f (snapshot r stride s) M N p row j.val :=
    inputStage_value a f r stride M N row p block s hsep
  have hc : ∀ j, normal.centered j=NativeParameters.centered red f (snapshot r stride s) M N p row j.val :=
    normalization_centered a red f r M N row eps p hblock first _ hv
  have hr : ∀ owner, normal.rstd owner=NativeParameters.rstd red f (snapshot r stride s) M N eps p row owner :=
    normalization_rstd a red f r M N row eps p hblock first _ hv
  have hread : TypedSource.ReadsEqual f r normal.memory s :=
    ((inputStage_frame a f r stride M N row p block s).trans
      (normalizationStage_frame a red f r stride M N row eps hblock first)).readsEqual hsep
  have hkeep : ∀ region, region=r.residualOut ∨ region=r.dropoutMask →
      ∀ offset, final.mem region offset=first.memory.mem region offset := by
    intro region hregion offset
    rw [affine_preserves a plan plan1 f r stride N row normal region offset
      (hkeepnames region hregion).1 (hkeepnames region hregion).2.1]
    exact normalization_preserves a red f r N row eps hblock first region offset
      (hkeepnames region hregion).2.2.1 (hkeepnames region hregion).2.2.2
  refine ⟨execute_frame a red plan plan1 f r stride M N row eps p hblock s,?_,?_,?_,?_,?_,?_,?_⟩
  · intro ho j hj
    change final.mem _ _=_
    rw [hkeep r.residualOut (Or.inl rfl)]
    exact input_residual_store a f r stride M N row p block s hsep ho j hj
  · intro hd hm j hj
    change final.mem _ _=_
    rw [hkeep r.dropoutMask (Or.inr rfl)]
    exact input_mask_store a f r stride M N row p block s hrow hmask hd hm j hj
  · intro hd hm hx j hj
    change final.mem _ _=_
    rw [hkeep r.dropoutMask (Or.inr rfl)]
    exact input_mask1_store a f r stride M N row p block s hsep hmask hd hm hx j hj
  · exact affine_y_store a red plan plan1 f r stride M N row eps p s normal hread h01 hc hr
  · exact affine_y1_store a red plan plan1 f r stride M N row eps p s normal hread hsep hc hr
  · change (affineStage a plan plan1 f r stride N row normal).mem r.rstd row=_
    rw [affine_preserves a plan plan1 f r stride N row normal r.rstd row h12 h02]
    exact normalization_rstd_store a red f r M N row eps p hblock first _ hv
  · intro hrms
    change (affineStage a plan plan1 f r stride N row normal).mem r.mean row=_
    rw [affine_preserves a plan plan1 f r stride N row normal r.mean row h13 h03]
    exact normalization_mean_store a red f r M N row eps p hblock first _ hv hrms h23

#axiomsClean execute_row_post
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
