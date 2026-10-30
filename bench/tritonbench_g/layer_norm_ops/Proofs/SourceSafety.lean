import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardWhole
import bench.tritonbench_g.layer_norm_ops.Proofs.ForwardRun
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety of the selected forward program and bounded backward loop.
The optional source branches determine which windows require backing. -/
namespace VeriTile.Bench.Repairs.LayerNormOps.SourceSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (p : ForwardParameters) (s : State α)
    (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : p.hasBias=Bool.true → ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias)
    (hres : p.hasResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualStride+i.val<bounds p.residual)
    (hy : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.yStride+i.val<bounds p.y)
    (hresout : p.storeResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualOutStride+i.val<bounds p.residualOut)
    (hm : p.rms=Bool.false → s.pids 0<bounds p.mean)
    (hr : s.pids 0<bounds p.rstd) :
    TraceSafe M bounds (forwardKernel p) s := by
  cases hR : p.rms <;> cases hA : p.hasResidual <;>
    cases hS : p.storeResidual <;> cases hB : p.hasBias
  all_goals
    simp [TraceSafe,forwardKernel,forwardSource,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      numeric,bop,numericLt,natLt,Region.cast,hR,hA,hS,hB,hx,hw,hb,hres,hy,hresout,hm,hr]
  all_goals repeat' first
    | apply And.intro
    | exact hx
    | exact hw
    | exact hy
    | exact hr
    | exact hb hB
    | exact hres hA
    | exact hresout hS
    | exact hm hR

#axiomsClean forward

private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (safe : traceSafeList M bounds a s) (ran : run M a s=some t)
    (next : traceSafeList M bounds b t) : traceSafeList M bounds (a++b) s := by
  apply (traceSafeList_append M bounds a b s).mpr
  refine ⟨safe,?_⟩
  intro u hu
  have eq : u=t := Option.some.inj (hu.symm.trans ran)
  subst u
  exact next

namespace Backward
open LayerNormOps.Backward

theorem prelude_safe (M : Algebra α) (p : BackwardParameters) (s : State α)
    (bounds : RegionName → Nat)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : p.recompute=Bool.true → p.hasBias=Bool.true →
      ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias) :
    traceSafeList M bounds (prelude p) s := by
  cases hD : p.hasDres <;> cases hS : p.storeDres <;>
    cases hB : p.hasBias <;> cases hY : p.recompute
  all_goals
    simp [prelude,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      integralFloorDiv,numericLt,natLt,Region.cast,hD,hS,hB,hY,hw,hb]
  all_goals repeat' first
    | apply And.intro
    | exact hw
    | exact hb hY hB

theorem body_safe (M : Algebra α) (p : BackwardParameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready p s t row a)
    (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → row*p.xStride+i.val<bounds p.x)
    (hdy : ∀ i : Fin p.block,i.val<p.columns → row*p.dyStride+i.val<bounds p.dy)
    (hm : p.rms=Bool.false → row*p.meanStride<bounds p.mean)
    (hr : row*p.rstdStride<bounds p.rstd)
    (hdres : p.hasDres=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      row*p.dresStride+i.val<bounds p.dres)
    (hy : p.recompute=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      row*p.yStride+i.val<bounds p.y)
    (hdresout : p.storeDres=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      row*p.dresInStride+i.val<bounds p.dresIn)
    (hdx : ∀ i : Fin p.block,i.val<p.columns → row*p.dxStride+i.val<bounds p.dx) :
    traceSafeList M bounds (body p) (t.setReg "row" .nat [] (fun _ => row)) := by
  cases hR : p.rms <;> cases hD : p.hasDres <;> cases hS : p.storeDres <;>
    cases hB : p.hasBias <;> cases hY : p.recompute
  all_goals
    simp [body,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      numeric,bop,numericLt,natLt,Region.cast,hR,hD,hS,hB,hY,
      h.mem,h.pid,h.first,h.last,h.cols,h.mask,h.dw,h.db,h.weight,h.bias,
      h.xp,h.yp,h.dxp,h.dyp,h.dres,h.dresIn,hx,hdy,hm,hr,hdres,hy,hdresout,hdx]
  all_goals repeat' first
    | apply And.intro
    | exact hx
    | exact hdy
    | exact hm hR
    | exact hr
    | exact hdres hD
    | exact hy hY
    | exact hdresout hS
    | exact hdx

theorem postlude_safe (M : Algebra α) (p : BackwardParameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready p s t row a)
    (bounds : RegionName → Nat)
    (hdw : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.columns+i.val<bounds p.dw)
    (hdb : p.hasBias=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.columns+i.val<bounds p.db) :
    traceSafeList M bounds (postlude p) t := by
  cases hB : p.hasBias
  all_goals
    simp [postlude,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      numeric,bop,numericLt,natLt,Region.cast,hB,h.mem,h.pid,h.cols,h.mask,h.dw,h.db,hdw,hdb]
  all_goals repeat' first
    | apply And.intro
    | exact hdw
    | exact hdb hB

structure Bounds (p : BackwardParameters) (pid : Nat) (bounds : RegionName → Nat) : Prop where
  weight : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight
  bias : p.recompute=Bool.true → p.hasBias=Bool.true →
    ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias
  x : ∀ row,pid*p.rowsPerProgram≤row → row<min ((pid+1)*p.rowsPerProgram) p.rows →
    ∀ i : Fin p.block,i.val<p.columns → row*p.xStride+i.val<bounds p.x
  dy : ∀ row,pid*p.rowsPerProgram≤row → row<min ((pid+1)*p.rowsPerProgram) p.rows →
    ∀ i : Fin p.block,i.val<p.columns → row*p.dyStride+i.val<bounds p.dy
  mean : p.rms=Bool.false → ∀ row,pid*p.rowsPerProgram≤row →
    row<min ((pid+1)*p.rowsPerProgram) p.rows → row*p.meanStride<bounds p.mean
  rstd : ∀ row,pid*p.rowsPerProgram≤row → row<min ((pid+1)*p.rowsPerProgram) p.rows →
    row*p.rstdStride<bounds p.rstd
  dres : p.hasDres=Bool.true → ∀ row,pid*p.rowsPerProgram≤row →
    row<min ((pid+1)*p.rowsPerProgram) p.rows →
    ∀ i : Fin p.block,i.val<p.columns → row*p.dresStride+i.val<bounds p.dres
  y : p.recompute=Bool.true → ∀ row,pid*p.rowsPerProgram≤row →
    row<min ((pid+1)*p.rowsPerProgram) p.rows →
    ∀ i : Fin p.block,i.val<p.columns → row*p.yStride+i.val<bounds p.y
  dresOut : p.storeDres=Bool.true → ∀ row,pid*p.rowsPerProgram≤row →
    row<min ((pid+1)*p.rowsPerProgram) p.rows →
    ∀ i : Fin p.block,i.val<p.columns → row*p.dresInStride+i.val<bounds p.dresIn
  dx : ∀ row,pid*p.rowsPerProgram≤row → row<min ((pid+1)*p.rowsPerProgram) p.rows →
    ∀ i : Fin p.block,i.val<p.columns → row*p.dxStride+i.val<bounds p.dx
  dw : ∀ i : Fin p.block,i.val<p.columns → pid*p.columns+i.val<bounds p.dw
  db : p.hasBias=Bool.true → ∀ i : Fin p.block,i.val<p.columns → pid*p.columns+i.val<bounds p.db

theorem range_safe (M : Algebra α) (p : BackwardParameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready p s t row a)
    (lower : start p s≤row) (bounds : RegionName → Nat) (hb : Bounds p (s.pids 0) bounds) :
    rangeSafe M bounds "row" row (stop p s) 1 (body p) t := by
  rw [rangeSafe_unfold,if_neg (by decide)]
  by_cases hi : row<stop p s
  · rw [if_pos hi]
    obtain ⟨u,hu,ready⟩ := body_run M p s t row a h
    refine ⟨body_safe M p s t row a h bounds (hb.x row lower hi)
      (hb.dy row lower hi) (fun hm => hb.mean hm row lower hi) (hb.rstd row lower hi)
      (fun hd => hb.dres hd row lower hi) (fun hy => hb.y hy row lower hi)
      (fun hd => hb.dresOut hd row lower hi) (hb.dx row lower hi),?_⟩
    rw [hu]
    exact range_safe M p s u (row+1) _ ready (by omega) bounds hb
  · rw [if_neg hi]
    trivial
termination_by stop p s-row
decreasing_by omega

theorem source_safe (M : Algebra α) (p : BackwardParameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p (s.pids 0) bounds) :
    TraceSafe M bounds (backwardKernel p) s := by
  obtain ⟨u,hu,hr⟩ := prelude_run M p s
  obtain ⟨v,hv,k,hk⟩ := range_run M p s u (start p s) (stop p s) (initial M p s) hr
  change traceSafeList _ _ (backwardKernel p).surfaceBody s
  rw [source_split]
  simp only [List.append_assoc]
  apply append_safe _ _ _ _ s u (prelude_safe M p s bounds hb.weight hb.bias) hu
  apply append_safe _ _ _ _ u v
  · apply traceSafeList_cons_intro
    · simpa [stepSafe_unfold,safeOp_unfold,evalOp_unfold,hr.first,hr.last] using
        range_safe M p s u (start p s) (initial M p s) hr (by rfl) bounds hb
    · intro t _
      exact traceSafeList_nil _ _ _
  · simp [run,step,evalOp_unfold,hr.first,hr.last,hv]
  · exact postlude_safe M p s v k _ hk bounds hb.dw hb.db

end Backward
#axiomsClean Backward.source_safe
end
end VeriTile.Bench.Repairs.LayerNormOps.SourceSafety
