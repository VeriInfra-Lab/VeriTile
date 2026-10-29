import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphExecution
import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult SourceGraph
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
attribute [local irreducible] treeValue

theorem prologue_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s : State Expr) :
    Safety.traceSafeList (engine rest plans) bounds (prologue p) s := by
  simp only [prologue, SourceGraph.source, VeriTile.Bench.Repairs.BatchedVecmatMult.source,
    VeriTile.Bench.Repairs.BatchedVecmatMult.kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_zero,List.getElem?_cons_succ,List.take,List.drop]
  simp [prologue, SourceGraph.source, VeriTile.Bench.Repairs.BatchedVecmatMult.source,
    VeriTile.Bench.Repairs.BatchedVecmatMult.kernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    numeric,numericLt,integralFloorDiv,ComputeDType.eraseDType,State.setReg,bop,Int.natCast_nonneg]

/-- Broadcast and carry arithmetic have no memory access, for every incoming
register state and opaque floating algebra. -/
theorem arithmetic_tail_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (t : State Expr) :
    Safety.traceSafeList (engine rest plans) bounds ((body p).drop 4) t := by
  simp only [body, SourceGraph.source, VeriTile.Bench.Repairs.BatchedVecmatMult.source,
    VeriTile.Bench.Repairs.BatchedVecmatMult.kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_zero,List.getElem?_cons_succ,List.take,List.drop]
  simp [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,evalOp_unfold]

theorem body_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State Expr) (c : Nat)
    (h : Invariant plans p s c t)
    (ha : ∀ (i : Fin p.BM) (k : Fin p.BK), aOffset p s i.val c k.val < bounds p.A)
    (hb : ∀ (i : Fin p.BM) (j : Fin p.BN) (k : Fin p.BK),
      bOffset p s i.val j.val c k.val < bounds p.B) :
    Safety.traceSafeList (engine rest plans) bounds (body p)
      (t.setReg "k_index" .int [] (fun _ => (c : Int))) := by
  rw [← List.take_append_drop 4 (body p),Safety.traceSafeList_append]
  refine ⟨?_,fun u _ => arithmetic_tail_safe rest plans p bounds u⟩
  simp only [body, SourceGraph.source, VeriTile.Bench.Repairs.BatchedVecmatMult.source,
    VeriTile.Bench.Repairs.BatchedVecmatMult.kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_zero,List.getElem?_cons_succ,List.take,List.drop]
  simp (config := { maxSteps := 700000 }) [
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    numeric,numericLt,integralFloorDiv,ComputeDType.eraseDType,State.setReg,bop,
    h.dimn,h.dimk,h.mi,h.ni,h.offsets,h.carry,h.blocks,h.memory,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  simpa only [aOffset,bOffset,row,col,TileShape.dropInsertedIndex,TileShape.insertAxis,
    ← Int.natCast_add,← Int.natCast_mul,Int.toNat_natCast] using
      And.intro ha (fun j i k => hb i j k)

theorem tail_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State Expr)
    (h : Invariant plans p s (p.K/p.BK) t)
    (ho : ∀ i : TileIndex [p.BM,p.BN], outputOffset p s i < bounds p.output) :
    Safety.traceSafeList (engine rest plans) bounds (tail p) t := by
  change Safety.traceSafeList (engine rest plans) bounds
    [.store .real [p.BM,p.BN] (.region p.output (.castIntToNat (.ref .int [p.BM,p.BN] "output_tile")))
      (.alg (.ref .real [p.BM,p.BN] "vecmat")) .none] t
  apply Safety.traceSafeList_cons_intro
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    evalOp_unfold,h.offsets,Region.cast,Int.toNat_natCast] using ho
  · intro _ _; exact Safety.traceSafeList_nil _ _ _

theorem source_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s : State Expr)
    (ha : ∀ c, c<p.K/p.BK → ∀ (i : Fin p.BM) (k : Fin p.BK),
      aOffset p s i.val c k.val < bounds p.A)
    (hb : ∀ c, c<p.K/p.BK → ∀ (i : Fin p.BM) (j : Fin p.BN) (k : Fin p.BK),
      bOffset p s i.val j.val c k.val < bounds p.B)
    (ho : ∀ i : TileIndex [p.BM,p.BN], outputOffset p s i < bounds p.output) :
    FP.Structural.TraceSafe (engine rest plans) bounds (source p) s := by
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
    apply Safety.rangeIntSafe_of_invariant (engine rest plans) bounds "k_index"
      ((p.K/p.BK : Nat) : Int) 1 (body p)
      (fun q current => 0 ≤ q ∧ Invariant plans p s q.toNat current)
    · intro q active current inv
      have cast : (q.toNat : Int) = q := Int.toNat_of_nonneg inv.1
      have bound : q.toNat < p.K/p.BK := by omega
      simpa only [cast] using
        body_safe rest plans p bounds s current q.toNat inv.2 (ha q.toNat bound) (hb q.toNat bound)
    · intro q active current inv next executed
      have cast : (q.toNat : Int) = q := Int.toNat_of_nonneg inv.1
      have ran := (bodyExecution rest plans p s current q.toNat inv.2).property
      simp only [cast] at ran
      have eq := Option.some.inj (ran.symm.trans executed)
      subst next
      refine ⟨by omega, ?_⟩
      have succ : (q + 1).toNat = q.toNat + 1 := by omega
      simpa only [Int.natCast_one,succ] using
        bodyInvariant rest plans p s current q.toNat inv.2
    · exact ⟨by omega,h⟩
  · intro t executed
    have hp := prologueInvariant rest plans p s
    obtain ⟨last,run,hi⟩ := loop_correct rest plans p s (p.K/p.BK) 0
      (prologueExecution rest plans p s).val (Nat.zero_le _) (by simp) hp
    simp only [Int.natCast_zero] at run
    simp only [run_append,(prologueExecution rest plans p s).property,
      Option.bind_some,FP.Structural.run,step,evalExpr,evalComputeOp,evalOp_unfold,
      hp.blocks,bind,Option.bind,run] at executed
    have eq := Option.some.inj executed
    subst t
    exact tail_safe rest plans p bounds s last hi ho

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelSafety
