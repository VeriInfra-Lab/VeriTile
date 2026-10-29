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
    simp [body,isRowLoop,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
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
