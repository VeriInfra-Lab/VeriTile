import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelFP
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelSafety
open VeriTile.Triton FP.Structural Source KernelFP
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Both metadata reads are unconditional, including when the split count is
zero. Their bounds follow the actually evaluated signed metadata expressions. -/
theorem prelude_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (length : s.pids 0 < bounds "lengths")
    (table : s.pids 0*p.strideBts+((position s).tdiv p.blockSize*p.strideBtb).toNat < bounds "table") :
    Safety.traceSafeList M bounds (prelude p) s := by
  simp (config := { maxSteps := 500000 }) [prelude, originalKernel, kernel, quotedOriginal, quotedCommuted,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericLt, bop, Region.cast, Option.bind,
    FP.Structural.eval_truncDiv, FP.Structural.eval_truncMod,
    Broadcast.leftIndex, Broadcast.rightIndex, State.setReg]
  refine ⟨length, ?_⟩
  simpa [position, integralFloorDiv, ← IntegerExpr.tdiv_formula,
    Broadcast.leftIndex, Broadcast.rightIndex, IntegerExpr.same_left,
    IntegerExpr.same_right, IntegerExpr.sameShape, bop] using table

/-- The complete K/V split body accesses four declared windows. The control
registers and frozen source regions are those maintained by actual execution. -/
theorem body_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (q : Nat)
    (controls : Controls p s t) (frozen : Frozen s t)
    (inputs : ∀ key (i : TileIndex [p.width]), sourceOffset p s.pids key q i < bounds (inputRegion key))
    (outputs : ∀ key (i : TileIndex [p.width]), (targetInt p s key q i).toNat < bounds (outputRegion key)) :
    Safety.traceSafeList M bounds (body Bool.false p)
      (t.setReg "split_x" .nat [] (fun _ => q)) := by
  rcases controls with ⟨hs, hh, hb, hslot, hl⟩
  rcases frozen with ⟨hk, hv⟩
  simp (config := { maxSteps := 700000 }) [body, kernel, quotedOriginal, quotedCommuted, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    step, evalExpr, evalComputeOp, evalOp_unfold, store, numeric, bop,
    ComputeDType.eraseDType, Region.cast, Option.bind, State.setReg,
    hs, hh, hb, hslot, hl, hk, hv, Broadcast.leftIndex, Broadcast.rightIndex]
  refine ⟨fun i => inputs Bool.true (i, ()), fun i => inputs Bool.false (i, ()),
    fun i => outputs Bool.true (i, ()), ?_⟩
  intro i
  simpa [targetInt, outputRegion, Int.natCast_add, Int.natCast_mul] using outputs Bool.false (i, ())

/-- The complete original source, including every static split, is safe under
exactly its two metadata bounds and its K/V input/output windows. -/
theorem trace {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (length : s.pids 0 < bounds "lengths")
    (table : s.pids 0*p.strideBts+((position s).tdiv p.blockSize*p.strideBtb).toNat < bounds "table")
    (inputs : ∀ q, q < p.dims/p.width → ∀ key (i : TileIndex [p.width]),
      sourceOffset p s.pids key q i < bounds (inputRegion key))
    (outputs : ∀ q, q < p.dims/p.width → ∀ key (i : TileIndex [p.width]),
      (targetInt p s key q i).toNat < bounds (outputRegion key)) :
    FP.Structural.TraceSafe M bounds (originalKernel p) s := by
  change Safety.traceSafeList M bounds (kernel Bool.false p).surfaceBody s
  rw [decomposition, Safety.traceSafeList_append]
  refine ⟨prelude_safe M p bounds s length table, ?_⟩
  intro t executed
  have eq : t = preludeState p s := Option.some.inj (executed.symm.trans (prelude_run M p s))
  subst t
  rw [Safety.traceSafeList_cons]
  refine ⟨?_, by intros; exact Safety.traceSafeList_nil _ _ _⟩
  rw [Safety.stepSafe_unfold]
  let P : Nat → State α → Prop := fun _ t => Controls p s t ∧ Frozen s t
  apply Safety.loopSafe_of_invariant M bounds "split_x" (p.dims/p.width) (body Bool.false p) P
  · intro q hq t ht
    exact body_safe M p bounds s t q ht.1 ht.2 (inputs q hq) (outputs q hq)
  · intro q _ t ht u run
    have eq : u = finalStep M p s q t :=
      Option.some.inj (run.symm.trans (body_run M Bool.false p s t q ht.1 ht.2))
    subst u
    exact ⟨finalStep_controls M p s t q ht.1, finalStep_frozen M p s t q ht.2⟩
  · refine ⟨?_, ⟨rfl, rfl⟩⟩
    constructor <;> simp [preludeState]

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean trace
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelSafety
