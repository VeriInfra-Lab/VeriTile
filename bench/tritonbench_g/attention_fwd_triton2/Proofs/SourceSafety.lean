import bench.tritonbench_g.attention_fwd_triton2.Proofs.Output
import VeriTile.Triton.Float.StructuralSafety

/-! Actual Structural access safety of the complete mixed-precision source.
The contexts and intermediate states are the existing source execution
witnesses, including all typed dot operations and the BF16 store. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole001 Whole001.Source
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  query : ∀ (i d : Fin 128),s.pids 0*128+i.val<p.length → d.val<96 →
    base p s p.q+(s.pids 0*128+i.val)*p.q.row+d.val*p.q.dim<bounds p.Q
  queryScale : s.pids 1*((p.length+127)/128)+s.pids 0<bounds p.QS
  key : ∀ row,p.length>row → ∀ (d : Fin 128) (j : Fin 64),j.val<p.length-row → d.val<96 →
    base p s p.k+d.val*p.k.dim+(row+j.val)*p.k.row<bounds p.K
  keyScale : ∀ row,row<p.length → s.pids 1*((p.length+63)/64)+row/64<bounds p.KS
  value : ∀ row,p.length>row → ∀ (j : Fin 64) (d : Fin 128),j.val<p.length-row → d.val<96 →
    base p s p.v+(row+j.val)*p.v.row+d.val*p.v.dim<bounds p.V
  output : ∀ (i d : Fin 128),s.pids 0*128+i.val<p.length → d.val<96 →
    base p s p.o+(s.pids 0*128+i.val)*p.o.row+d.val*p.o.dim<bounds p.Out

private theorem source_offset (ph heads batch head pm i row d dim : Nat) :
    ((ph:Int)/(heads:Int)*(batch:Int)+((ph:Int)-(heads:Int)*((ph:Int)/(heads:Int)))*(head:Int)+
      ((pm:Int)*128+(i:Int))*(row:Int)+(d:Int)*(dim:Int)).toNat =
      ph/heads*batch+ph%heads*head+(pm*128+i)*row+d*dim := by
  have hm : (ph:Int)-(heads:Int)*((ph:Int)/(heads:Int))=((ph%heads:Nat):Int) := by
    rw [←Int.emod_def,←Int.natCast_emod]
  rw [hm]
  norm_cast

theorem entry_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (bounds : RegionName → Nat)
    (s : State α) (hb : Bounds p s bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).before s := by
  simp (config := {maxSteps := 200000}) [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
  refine ⟨?_,hb.queryScale⟩
  intro i d hr hd
  have hrow : (s.pids 0:Int)*128+(i.val:Int)<(p.length:Int) := hr
  have hn : s.pids 0*128+i.val<p.length := by omega
  simpa only [source_offset,base] using hb.query i d hn hd

theorem body_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (bounds : RegionName → Nat)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) (inside : row<p.length) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).body
      (current.setReg "start_n" .nat [] (fun _=>row)) := by
  simp (config := {maxSteps := 200000}) [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.rows,ctx.cols,ctx.query,ctx.scale,ctx.keyPointer,ctx.valuePointer,
    ctx.scalePointer,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value]
  exact ⟨hb.key row inside,hb.keyScale row inside,hb.value row inside⟩

theorem post_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (bounds : RegionName → Nat)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).after current := by
  simp (config := {maxSteps := 200000}) [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.rows,ctx.outputPointer,ctx.normalizer,ctx.value]
  refine ⟨?_,by split <;> trivial⟩
  intro i d hr hd
  have hrow : ((initial.pids 0*128+i.val:Nat):Int)<(p.length:Int) := hr
  have hn : initial.pids 0*128+i.val<p.length := by omega
  exact hb.output i d hn hd

end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.SourceSafety
