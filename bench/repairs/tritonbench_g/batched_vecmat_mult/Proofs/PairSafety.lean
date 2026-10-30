import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.PairComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.BlockFirst
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
  simp [prologue, source, VeriTile.Bench.Repairs.BatchedVecmatMult.blockFirst, TypedSource.blockFirst, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,numeric,integralFloorDiv,ComputeDType.eraseDType,State.setReg,bop]

theorem body_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State Expr) (c : Nat)
    (h : Invariant plans p s c t)
    (ha : ∀ (i : Fin p.BM) (k : Fin p.BK), aOffset p s i.val c k.val < bounds p.A)
    (hb : ∀ (i : Fin p.BM) (j : Fin p.BN) (k : Fin p.BK),
      bOffset p s i.val j.val c k.val < bounds p.B) :
    Safety.traceSafeList (engine rest plans) bounds (body p)
      (t.setReg "k_index" .nat [] (fun _ => c)) := by
  simp (config := { maxSteps := 700000 }) [body,source,VeriTile.Bench.Repairs.BatchedVecmatMult.blockFirst,TypedSource.blockFirst,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,numeric,integralFloorDiv,ComputeDType.eraseDType,State.setReg,bop,
    h.mi,h.ni,h.offsets,h.carry,h.blocks,h.memory,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  exact ⟨fun i k => ha i k, fun j i k => hb i j k⟩

theorem tail_safe (rest : Algebra Expr) (plans : PlanSelector) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State Expr)
    (h : Invariant plans p s (p.K/p.BK) t)
    (ho : ∀ i : TileIndex [p.BM,p.BN], outputOffset p s i < bounds p.output) :
    Safety.traceSafeList (engine rest plans) bounds (tail p) t := by
  change Safety.traceSafeList (engine rest plans) bounds
    [.store .real [p.BM,p.BN] (.region p.output (.ref .nat [p.BM,p.BN] "output_tile"))
      (.alg (.ref .real [p.BM,p.BN] "vecmat")) .none] t
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
    apply Safety.rangeSafe_of_invariant (engine rest plans) bounds "k_index" (p.K/p.BK) 1
      (body p) (Invariant plans p s)
    · intro c active current inv
      exact body_safe rest plans p bounds s current c inv (ha c active) (hb c active)
    · intro c active current inv next executed
      have eq := Option.some.inj ((bodyExecution rest plans p s current c inv).property.symm.trans executed)
      subst next
      exact bodyInvariant rest plans p s current c inv
    · exact h
  · intro t executed
    have hp := prologueInvariant rest plans p s
    obtain ⟨last,run,hi⟩ := loop_correct rest plans p s (p.K/p.BK) 0
      (prologueExecution rest plans p s).val (Nat.zero_le _) (by simp) hp
    simp only [run_append,(prologueExecution rest plans p s).property,
      Option.bind_some,FP.Structural.run,step,evalExpr,evalComputeOp,evalOp_unfold,
      hp.blocks,bind,Option.bind,run] at executed
    have eq := Option.some.inj executed
    subst t
    exact tail_safe rest plans p bounds s last hi ho

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.BlockFirst
