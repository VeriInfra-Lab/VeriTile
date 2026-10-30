import bench.tritonbench_g.bmm_optimized.Proofs.GraphRun

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
open VeriTile.Triton FP.Structural BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem singleton_safe (M : Algebra α) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

/-- The actual source prelude only forms pointers and scalar registers. -/
theorem prelude_safe (p : Scalar α) (rest : Algebra α) (cfg : Parameters)
    (bounds : RegionName → Nat) (s : State (Node α)) :
    Safety.traceSafeList (engine p rest) bounds (prelude cfg) s := by
  cases dm : cfg.divM <;> cases dn : cfg.divN <;> cases dk : cfg.divK
  all_goals by_cases hg : cfg.GROUP=1
  all_goals by_cases ht : s.numPids 0 <
    (s.pids 0+s.pids 1*s.numPids 0)/(s.numPids 1*cfg.GROUP)*cfg.GROUP+cfg.GROUP
  all_goals
    simp [prelude,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
      dm,dn,dk,hg,ht,singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,
      Safety.safeCompute,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
      FP.Structural.step,FP.Structural.run,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod]

/-- Actual evaluated input masks and pointers are bounded at each reached
iteration. Bounds are physical window facts, not numerical assumptions. -/
theorem body_safe (p : Scalar α) (rest : Algebra α) (cfg : Parameters)
    (bounds : RegionName → Nat) (initial current : State (Node α)) (k : Nat)
    (h : Invariant (scalar p) cfg initial k current)
    (ha : ∀ i : AIndex cfg,
      (cfg.divM=Bool.true ∨ row cfg initial i.1<cfg.M) ∧
        (cfg.divK=Bool.true ∨ inner cfg k i.2.1<cfg.K) →
      aOffset cfg initial k i<bounds cfg.A)
    (hb : ∀ i : BIndex cfg,
      (cfg.divK=Bool.true ∨ inner cfg k i.1<cfg.K) ∧
        (cfg.divN=Bool.true ∨ column cfg initial i.2.1<cfg.N) →
      bOffset cfg initial k i<bounds cfg.B) :
    Safety.traceSafeList (engine p rest) bounds (body cfg) current := by
  have hm:=h.maskM
  have hn:=h.maskN
  cases dm : cfg.divM <;> cases dn : cfg.divN <;> cases dk : cfg.divK
  all_goals
    simp only [dm,dn,dk,Bool.false_eq_true,false_or,true_or,true_and,and_true,
      false_implies,true_implies] at hm hn ha hb
    simp [body,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
      dm,dn,dk,singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,
      Safety.safeCompute,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
      h.memory,h.aptrs,h.bptrs,h.offsk,h.acc,hm,hn,engine,algebra,ha,hb]
    all_goals first
      | exact ⟨fun i j rest hr hk=>ha (i,j,rest) ⟨hr,hk⟩,
          fun i j rest hk hn=>hb (i,j,rest) ⟨hk,hn⟩⟩
      | exact ⟨fun i j rest hr=>ha (i,j,rest) hr,
          fun i j rest hn=>hb (i,j,rest) hn⟩
      | exact ⟨fun i j rest hr hk=>ha (i,j,rest) ⟨hr,hk⟩,
          fun i j rest hk=>hb (i,j,rest) hk⟩
      | exact ⟨fun i j rest hr=>ha (i,j,rest) hr,
          fun i j rest=>hb (i,j,rest)⟩
      | exact ⟨fun i j rest hk=>ha (i,j,rest) hk,
          fun i j rest hk hn=>hb (i,j,rest) ⟨hk,hn⟩⟩
      | exact ⟨fun i j rest=>ha (i,j,rest),
          fun i j rest hn=>hb (i,j,rest) hn⟩
      | exact ⟨fun i j rest hk=>ha (i,j,rest) hk,
          fun i j rest hk=>hb (i,j,rest) hk⟩
      | exact ⟨fun i j rest=>ha (i,j,rest),fun i j rest=>hb (i,j,rest)⟩
      | exact fun i j rest hr=>ha (i,j,rest) hr
      | exact fun i j rest hn=>hb (i,j,rest) hn

/-- The actual store, including every source mask branch, is bounded. -/
theorem epilogue_safe (p : Scalar α) (rest : Algebra α) (cfg : Parameters)
    (bounds : RegionName → Nat) (initial current : State (Node α))
    (h : Invariant (scalar p) cfg initial (iterations cfg) current)
    (ho : ∀ i : CIndex cfg,active cfg initial i → oOffset cfg initial i<bounds cfg.O) :
    Safety.traceSafeList (engine p rest) bounds (epilogue cfg) current := by
  have hm:=h.maskM
  have hn:=h.maskN
  cases dm : cfg.divM <;> cases dn : cfg.divN <;> cases dk : cfg.divK
  all_goals
    simp only [active,dm,dn,Bool.false_eq_true,false_or,true_or,true_and,and_true,
      false_implies,true_implies] at hm hn ho
    simp [epilogue,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
      dm,dn,dk,singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,
      Safety.safeCompute,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
      h.memory,h.optrs,h.acc,hm,hn,engine,algebra,ho]
    all_goals first
      | exact fun i j rest hr hn=>ho (i,j,rest) ⟨hr,hn⟩
      | exact fun i j rest hr=>ho (i,j,rest) hr
      | exact fun i j rest hn=>ho (i,j,rest) hn
      | exact fun i j rest=>ho (i,j,rest)

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
