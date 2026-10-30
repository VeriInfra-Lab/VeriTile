import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.FPComplete
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IONativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Host FPValue
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

def position (s : State A) : Math.Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩

theorem entry_safe (M : Algebra A) (c : Shape) (scale : ℝ) (s : State A)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine M) bounds (Typed.parts c scale).before s := by
  simp [Typed.parts,kernel,ChunkGlaSimpleMinDotTile.source,fp32Original,ComputeKernel.surfaceBody,
    Source.dynamicRange?,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,numericLe,natLe,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]

theorem body_safe (M : Algebra A) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (initial current : State A) (block : Nat) (acc : Acc A c)
    (ctx : Context c initial current) (ha : HasAcc c acc current)
    (bounds : RegionName → Nat)
    (hb : Safe.StepBounds (layout c) (position initial) regions block bounds) :
    Safety.traceSafeList (engine M) bounds (Typed.parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => block)) := by
  have ho := ha.1
  have hs := ha.2
  simp (maxSteps := 500000) [Typed.parts,kernel,ChunkGlaSimpleMinDotTile.source,fp32Original,
    ComputeKernel.surfaceBody,Source.dynamicRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,ctx.bh,ctx.it,ctx.rows,ctx.cols,ctx.mask,ctx.memory,ho,hs,
    d.qh_eq,d.qk_eq]
  refine ⟨?_,?_,?_⟩
  · intro i j hi hj
    simpa [layout,regions,position,FPValue.row,FPValue.key,Nat.mul_one] using hb.query i j ⟨hi,hj⟩
  · intro i j hi hj
    simpa [layout,regions,position,FPValue.row,FPValue.key,Nat.mul_one] using hb.key i j ⟨hi,hj⟩
  · intro i j hi hj
    simpa [layout,regions,position,FPValue.column,FPValue.key,Nat.mul_one] using hb.history i j ⟨hi,hj⟩

theorem finish_safe (M : Algebra A) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (initial current : State A) (ctx : Context c initial current)
    (ha : HasAcc c (accumulated M c d initial) current) (bounds : RegionName → Nat)
    (hb : Safe.TailBounds (layout c) (position initial) regions bounds) :
    Safety.traceSafeList (engine M) bounds (Typed.parts c scale).after current := by
  have ho := ha.1
  have hs := ha.2
  simp (maxSteps := 500000) [Typed.parts,kernel,ChunkGlaSimpleMinDotTile.source,fp32Original,
    ComputeKernel.surfaceBody,Source.dynamicRange?,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,ctx.bh,ctx.it,ctx.rows,ctx.cols,ctx.mask,ctx.memory,ho,hs,d.sv_eq]
  refine ⟨?_,?_,?_,?_⟩
  · intro i hi
    simpa [layout,regions,position,FPValue.row,Math.row,Math.gateAddress] using hb.gate i hi
  · intro i j hi hj
    simpa [layout,regions,position,FPValue.row,FPValue.column,Nat.mul_one] using hb.value i j ⟨hi,hj⟩
  · intro i j hi hj
    simpa [layout,regions,position,FPValue.row,FPValue.column,Nat.mul_one] using hb.output i j ⟨hi,hj⟩
  · split <;> trivial

theorem range_safe (M : Algebra A) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (initial : State A) (bounds : RegionName → Nat)
    (hb : Safe.Bounds (layout c) (position initial) regions bounds)
    (cur : Nat) (current : State A) (acc : Acc A c)
    (ctx : Context c initial current) (ha : HasAcc c acc current) :
    Safety.rangeSafe (engine M) bounds "i_k" cur ((c.K+c.BK-1)/c.BK) 1
      (Typed.parts c scale).body current := by
  rw [Safety.rangeSafe_unfold]
  simp only [show (1:Nat)≠0 by decide, ↓reduceIte]
  split
  · rename_i hn
    refine ⟨body_safe M c scale d initial current cur acc ctx ha bounds (hb.loop cur hn), ?_⟩
    rw [(bodyExecution M c scale d initial current cur acc ctx ha).property]
    exact range_safe M c scale d initial bounds hb (cur+1) _ _
      (bodyContext M c scale d initial current cur acc ctx ha).1
      (bodyContext M c scale d initial current cur acc ctx ha).2
  · trivial
termination_by ((c.K+c.BK-1)/c.BK)-cur
decreasing_by omega

theorem source_safe (M : Algebra A) (c : Shape) (scale : ℝ) (d : Typed.Dots M c)
    (initial : State A) (bounds : RegionName → Nat)
    (hb : Safe.Bounds (layout c) (position initial) regions bounds) :
    FP.Structural.TraceSafe (engine M) bounds (kernel c .fp32 .fp32 .fp32 .fp32 scale) initial := by
  change Safety.traceSafeList (engine M) bounds _ initial
  rw [Typed.source_parts]
  simp only [List.append_assoc, List.singleton_append]
  rw [Safety.traceSafeList_append]
  refine ⟨entry_safe M c scale initial bounds, ?_⟩
  intro start he
  have heq := Option.some.inj ((entryExecution M c scale initial).property.symm.trans he)
  subst start
  let ctx := (entryContext M c scale initial).1
  let ha := (entryContext M c scale initial).2
  obtain ⟨mid, hm, hc, hv⟩ := loopExecution M c scale d initial _ ctx ha
  apply Safety.traceSafeList_cons_of_step (t := mid)
  · change Safety.stepSafe (engine M) bounds (.forRangeDyn "i_k" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat c.K)
        (.constNat c.BK)) (.constNat 1)) (.constNat c.BK)) (.constNat 1)
      (Typed.parts c scale).body) _
    simp only [Safety.stepSafe_unfold, Safety.safeOp_unfold, evalOp_unfold,
      numeric, bop, Option.bind, Option.bind_some, true_and]
    exact range_safe M c scale d initial bounds hb 0 _ _ ctx ha
  · change step (engine M) (.forRangeDyn "i_k" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat c.K)
        (.constNat c.BK)) (.constNat 1)) (.constNat c.BK)) (.constNat 1)
      (Typed.parts c scale).body) _ = some mid
    simp only [step, evalOp_unfold, numeric, bop, Option.bind, Option.bind_some]
    exact hm
  · exact finish_safe M c scale d initial mid hc hv bounds hb.tail

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IONativeSafety
