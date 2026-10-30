import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.SourceSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem entry (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine (withDot M D)) bounds (fwdParts p).before s := by
  simp [fwdParts,forward,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]

theorem body (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (bounds : RegionName → Nat) (backed : ForwardBounds p pid bounds)
    (input : Memory α) (s : State α) (block : Nat) (carry : Matrix α p.D p.BM)
    (ctx : FwdReady p pid input s) (rows : FwdRows p block s) (hc : FwdCarry p carry s) :
    Safety.traceSafeList (engine (withDot M D)) bounds (fwdParts p).body
      (s.setReg "i" .nat [] (fun _ => block)) := by
  have hq := ctx.inputs "query" (by simp)
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  simp only [FwdRows] at rows
  simp only [FwdCarry] at hc
  simp [fwdParts,forward,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    ctx.query,ctx.key,ctx.value,ctx.output,ctx.causal,rows,hc,hq,hk,hv,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  refine ⟨?_,?_,?_,?_⟩
  · intro t a active
    have := backed.query (block*p.BT+t.val) a.val active a.isLt
    omega
  · intro a t active
    have := backed.key (block*p.BT+t.val) a.val active a.isLt
    omega
  · intro t c active
    have := backed.value (block*p.BT+t.val) c.val active c.isLt
    omega
  · intro t c active
    have := backed.output (block*p.BT+t.val) c.val active c.isLt
    omega

theorem loop (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (bounds : RegionName → Nat) (backed : ForwardBounds p pid bounds)
    (input : Memory α) (s : State α) (ctx : FwdReady p pid input s)
    (rows : FwdRows p 0 s) (hc : FwdCarry p (fun _ => zero M) s) :
    Safety.rangeSafe (engine (withDot M D)) bounds "i" 0 p.NT 1 (fwdParts p).body s := by
  let I := fun n (t : State α) => FwdReady p pid input t ∧ FwdRows p n t ∧
    FwdCarry p (fwdCarry M D p pid input n) t
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ I
  · intro n hn t hi
    exact body M D p pid bounds backed input t n _ hi.1 hi.2.1 hi.2.2
  · intro n hn t hi u ran
    have same : u = (fwdBodyExecution M D p pid input t n _ hi.1 hi.2.1 hi.2.2).val :=
      Option.some.inj (ran.symm.trans (fwdBodyExecution M D p pid input t n _ hi.1 hi.2.1 hi.2.2).property)
    subst u
    exact fwdBody_ready M D p pid input t n _ hi.1 hi.2.1 hi.2.2
  · exact ⟨ctx,rows,hc⟩

theorem program (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (backed : ForwardBounds p ⟨s.pids 0,s.pids 1⟩ bounds) :
    TraceSafe (engine (withDot M D)) bounds (forward p) s := by
  let start := fwdEntry M D p s
  have he := fwdEntry_ready M D p s
  change Safety.traceSafeList _ _ (forward p).surfaceBody s
  rw [forward_parts,Safety.traceSafeList_append]
  refine ⟨entry M D p bounds s,?_⟩
  intro t ran
  have same : t = start.val := Option.some.inj (ran.symm.trans start.property)
  subst t
  have safe := loop M D p ⟨s.pids 0,s.pids 1⟩ bounds backed s.mem start.val he.1 he.2.1 he.2.2
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeOp_unfold,
    evalOp_unfold,true_and]
  refine ⟨safe,?_⟩
  cases step (engine (withDot M D))
    (.forRangeDyn "i" (.constNat 0) (.constNat p.NT) (.constNat 1) (fwdParts p).body) start.val <;> trivial

#axiomsClean program

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
