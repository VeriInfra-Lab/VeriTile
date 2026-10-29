import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetySteps

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety Reused FPEngine FPContext
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (bounds : RegionName → Nat) : Prop where
  key : ∀ (d : Fin D) (j : Fin B), p.evenN=Bool.true ∨ s.pids 0*B+j.val<p.keyContext →
    baseK p s+d.val*p.keyDim+(s.pids 0*B+j.val)*p.keyRow < bounds p.K
  query : ∀ row, row%B=0 → row<p.queryContext → ∀ (i : Fin B) (d : Fin D),
    p.evenM=Bool.true ∨ row+i.val<p.queryContext →
      baseQ p s+(row+i.val)*p.queryRow+d.val*p.queryDim < bounds p.Q
  normalizer : ∀ row, row<p.queryContext → baseM p s+row < bounds p.M
  output : ∀ i : Fin B, s.pids 0*B+i.val<p.keyContext →
    FPOutput.address B p s (i,()) < bounds p.Out

theorem prelude (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds B D p s bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts B D p).before s := by
  rw [←List.take_append_drop 12 (parts B D p).before,traceSafeList_append]
  refine ⟨?_,fun next _ => prelude_tail M dot B D p next bounds⟩
  by_cases hn : p.evenN=Bool.true
  all_goals
    simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,hn,
      ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2,safeAdvance2]
  all_goals simp [safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,BlockPtr.inBounds_2d_offsets,
      BlockPtr.address_2d_offsets,TileShape.indexToList]
  · intro d j
    exact hb.key d j (Or.inl hn)
  · intro d j hj
    exact hb.key d j (Or.inr hj)

theorem body (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) (bounds : RegionName → Nat)
    (hb : Bounds B D p initial bounds) (hrow : row%B=0) (hstop : row<p.queryContext) :
    traceSafeList (engine (withDot M dot)) bounds (parts B D p).body
      (current.setReg "start_m" .nat [] (fun _ => row)) := by
  rw [←List.take_append_drop 3 (parts B D p).body,traceSafeList_append]
  refine ⟨?_,fun next _ => body_tail M dot B D p next bounds⟩
  by_cases hm : p.evenM=Bool.true
  all_goals
    simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,hm,
      ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2,safeAdvance2]
  all_goals simp [safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,queryPointer,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      ctx.memory,ctx.startN,ctx.query,ctx.normalizer,ctx.key,ctx.scale,ctx.output]
  all_goals
    constructor
    · intro i d
      first
      | exact hb.query row hrow hstop i d (Or.inl hm)
      | intro hi; exact hb.query row hrow hstop i d (Or.inr hi)
    · intro i hi
      simpa [Nat.add_assoc] using hb.normalizer (row+i.val) hi

theorem output (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) (bounds : RegionName → Nat)
    (hb : Bounds B D p initial bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts B D p).after current := by
  simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
    safeBlockPtr2,safeAdvance2,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.startN,ctx.batch,ctx.head,ctx.output,FP.Structural.store]
  intro i hi
  simpa [FPOutput.address,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using
    hb.output i (by simpa [Nat.add_comm] using hi)

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
