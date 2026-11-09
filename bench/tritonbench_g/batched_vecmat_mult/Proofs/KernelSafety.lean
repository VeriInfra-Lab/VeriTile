import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphExecution
import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult SourceGraph
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
attribute [local irreducible] treeValue

theorem prologue_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s : State Expr) :
    Safety.traceSafeList (engine rest plans) bounds (prologue p) s := by
  simp [prologue, SourceGraph.source, source, TypedSource.original, VeriTile.Bench.Repairs.BatchedVecmatMult.sourceKernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold,
    FP.Structural.run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,numeric,numericLt,integralFloorDiv,ComputeDType.eraseDType,State.setReg,bop]

theorem body_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State Expr) (c : Nat)
    (h : Invariant plans p s c t)
    (ha : ∀ (i : Fin p.BM) (k : Fin p.BK), aOffset p s i.val c k.val < bounds p.A)
    (hb : ∀ (i : Fin p.BM) (j : Fin p.BN) (k : Fin p.BK),
      bOffset p s i.val j.val c k.val < bounds p.B) :
    Safety.traceSafeList (engine rest plans) bounds (body p)
      (t.setReg "k_index" .int [] (fun _ => (c:Int))) := by
  simp (config := { maxSteps := 700000 }) [body,SourceGraph.source,source,TypedSource.original,VeriTile.Bench.Repairs.BatchedVecmatMult.sourceKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    FP.Structural.run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,numeric,numericLt,integralFloorDiv,ComputeDType.eraseDType,State.setReg,bop,
    h.mi,h.ni,h.offsets,h.carry,h.blocks,h.memory,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,TileShape.insertAxis]
  exact ⟨fun i k => ha i k, fun j i k => hb i j k⟩

theorem tail_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State Expr)
    (h : Invariant plans p s (p.K/p.BK) t)
    (ho : ∀ i : TileIndex [p.BM,p.BN], outputOffset p s i < bounds p.output) :
    Safety.traceSafeList (engine rest plans) bounds (tail p) t := by
  change Safety.traceSafeList (engine rest plans) bounds
    [.store .fp32 [p.BM,p.BN]
      (.region p.output (.castIntToNat (.ref .int [p.BM,p.BN] "output_tile")))
      (.alg (.castFloat .real .fp32 (.withPrecision .real .fp32 (.ref .real [p.BM,p.BN] "vecmat")))) .none] t
  apply Safety.traceSafeList_cons_intro
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    evalOp_unfold,h.offsets,Region.cast] using ho
  · intro _ _; exact Safety.traceSafeList_nil _ _ _

theorem source_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s : State Expr)
    (ha : ∀ c, c<p.K/p.BK → ∀ (i : Fin p.BM) (k : Fin p.BK),
      aOffset p s i.val c k.val < bounds p.A)
    (hb : ∀ c, c<p.K/p.BK → ∀ (i : Fin p.BM) (j : Fin p.BN) (k : Fin p.BK),
      bOffset p s i.val j.val c k.val < bounds p.B)
    (ho : ∀ i : TileIndex [p.BM,p.BN], outputOffset p s i < bounds p.output) :
    FP.Structural.TraceSafe (engine rest plans) bounds (source p) s := by
  unfold FP.Structural.TraceSafe
  change Safety.traceSafeList (engine rest plans) bounds (SourceGraph.source p).surfaceBody s
  rw [source_split,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨prologue_safe rest plans p bounds s, ?_⟩, ?_⟩
  · intro t run
    have eq := Option.some.inj ((prologueExecution rest plans p s).property.symm.trans run)
    subst t
    have h := prologueInvariant rest plans p s
    rw [Safety.traceSafeList_cons]
    refine ⟨?_, by intros; exact Safety.traceSafeList_nil _ _ _⟩
    simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,h.blocks,
      Option.bind_some,forall_const,true_and]
    apply Safety.rangeIntSafe_of_invariant (engine rest plans) bounds "k_index" ((p.K/p.BK:Nat):Int) 1
      (body p) (fun q t => 0 ≤ q ∧ Invariant plans p s q.toNat t)
    · intro q active current inv
      have converted : (q.toNat:Int)=q := by have := inv.1; omega
      have inside : q.toNat < p.K/p.BK := by omega
      simpa only [converted] using body_safe rest plans p bounds s current q.toNat inv.2
        (ha q.toNat inside) (hb q.toNat inside)
    · intro q active current inv next executed
      have converted : (q.toNat:Int)=q := by have := inv.1; omega
      have ran := (bodyExecution rest plans p s current q.toNat inv.2).property
      have actual : FP.Structural.run (engine rest plans) (body p)
          (current.setReg "k_index" .int [] (fun _ => (q.toNat:Int))) = some next := by
        simpa only [converted] using executed
      have eq := Option.some.inj (ran.symm.trans actual)
      subst next
      refine ⟨by have := inv.1; omega, ?_⟩
      have successor : (q+1).toNat=q.toNat+1 := by have := inv.1; omega
      simpa only [Nat.cast_one,successor] using bodyInvariant rest plans p s current q.toNat inv.2
    · exact ⟨by decide,h⟩
  · intro t executed
    have hp := prologueInvariant rest plans p s
    obtain ⟨last,run,hi⟩ := loop_correct rest plans p s (p.K/p.BK) 0
      (prologueExecution rest plans p s).val (Nat.zero_le _) (by simp) hp
    simp only [Nat.cast_zero] at run
    simp only [run_append,(prologueExecution rest plans p s).property,
      Option.bind_some,FP.Structural.run,step,evalExpr,evalComputeOp,evalOp_unfold,
      hp.blocks,bind,Option.bind,run] at executed
    have eq := Option.some.inj executed
    subst t
    exact tail_safe rest plans p bounds s last hi ho

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelSafety
