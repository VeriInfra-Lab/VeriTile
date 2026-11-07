import bench.tritonbench_g.flash_decode2_phi.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafety

/-! Actual source trace safety for both shared flash-decode source variants.
Proofs follow the existing loop invariant, actual evaluator states and metadata
loop bound. No floating facts or output/callback safety premise is introduced.
-/
namespace VeriTile.Bench.Repairs.FlashDecode2.AccessSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prologue_safe (A : Algebra α) (s : State α) (H S D Q BD : Nat)
    (bounds : RegionName → Nat) (length : s.pids 0 < bounds "L") :
    Safety.traceSafeList A bounds (prologue H S D Q BD) s := by
  simp [prologue, phiKernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericLt, numericLe, natLt, natLe, bop, Region.cast,
    Option.bind, ofFloat, toFloat, ComputeDType.eraseDType]
  exact length

theorem body_safe (A : Algebra α) (s q : State α) (masked : Bool)
    (H S D Q BD j : Nat) (I : Inv A s masked H S D Q BD j q)
    (bounds : RegionName → Nat)
    (mid : ∀ k : TileIndex [BD], active masked D k → voff s H S D j k < bounds "M")
    (log : eoff s H S j < bounds "E") :
    Safety.traceSafeList A bounds (body masked H S D Q BD)
      (q.setReg "block_seq_n" .int [] (fun _ => (j : Int))) := by
  rcases I with ⟨hm,hp,hb,hh,hl,hs,hv,he,hmax,hsum,ha⟩
  cases masked <;>
    simp [body, kernel, phiKernel, llamaKernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
      numeric, numericLt, numericLe, natLt, natLe, bop, Region.cast, Option.bind,
      ofFloat, toFloat, ComputeDType.eraseDType,
      hb,hh,hl,hs,hv,he,hmax,hsum,ha,hm,hp]
  · simp only [←Int.natCast_mul, ←Int.natCast_add, Int.toNat_natCast]
    constructor
    · intro k b
      simpa [voff, lane] using mid (k, b) (Or.inl rfl)
    · simpa [eoff] using log
  · simp only [←Int.natCast_mul, ←Int.natCast_add, Int.toNat_natCast]
    constructor
    · intro k b hk
      simpa [voff, lane] using mid (k, b) (Or.inr hk)
    · simpa [eoff] using log

theorem epilogue_safe (A : Algebra α) (s q : State α) (masked : Bool)
    (H S D Q BD : Nat) (I : Inv A s masked H S D Q BD (count s Q) q)
    (bounds : RegionName → Nat)
    (out : ∀ k : TileIndex [BD], active masked D k → ooff s H D k < bounds "O") :
    Safety.traceSafeList A bounds (epilogue masked H S D Q BD) q := by
  rcases I with ⟨hm,hp,hb,hh,hl,hs,hv,he,hmax,hsum,ha⟩
  cases masked <;>
    simp [epilogue, kernel, phiKernel, llamaKernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
      numeric, numericLt, numericLe, natLt, natLe, bop, Region.cast, Option.bind,
      ofFloat, toFloat, ComputeDType.eraseDType, store,
      hb,hh,hl,hs,hv,he,hmax,hsum,ha,hm,hp]
  · intro k b
    exact out (k, b) (Or.inl rfl)
  · intro k b hk
    exact out (k, b) (Or.inr hk)

theorem range_safe (A : Algebra α) (s q : State α) (masked : Bool)
    (H S D Q BD start : Nat) (I : Inv A s masked H S D Q BD start q)
    (bounds : RegionName → Nat)
    (mid : ∀ j, j < count s Q → ∀ k : TileIndex [BD],
      active masked D k → voff s H S D j k < bounds "M")
    (log : ∀ j, j < count s Q → eoff s H S j < bounds "E") :
    Safety.rangeIntSafe A bounds "block_seq_n" start (count s Q) 1 (body masked H S D Q BD) q := by
  rw [Safety.rangeIntSafe_unfold]
  simp only [Nat.one_ne_zero, if_false]
  by_cases inside : start < count s Q
  · have hi : (start : Int) < count s Q := by exact_mod_cast inside
    simp only [hi, if_true]
    refine ⟨body_safe A s q masked H S D Q BD start I bounds (mid start inside) (log start inside), ?_⟩
    obtain ⟨next, ran, invariant⟩ := body_run A s q masked H S D Q BD start I
    rw [ran]
    simpa only [Nat.cast_add, Nat.cast_one] using
      range_safe A s next masked H S D Q BD (start + 1) invariant bounds mid log
  · have hi : ¬ (start : Int) < count s Q := by exact_mod_cast inside
    simp [hi]

termination_by count s Q - start

/-- Actual complete source accesses obey the supplied independent coordinate
bounds, through every dynamically selected loop iteration. -/
theorem source_safe (A : Algebra α) (s : State α) (masked : Bool)
    (H S D Q BD : Nat) (bounds : RegionName → Nat)
    (length : s.pids 0 < bounds "L")
    (mid : ∀ j, j < count s Q → ∀ k : TileIndex [BD],
      active masked D k → voff s H S D j k < bounds "M")
    (log : ∀ j, j < count s Q → eoff s H S j < bounds "E")
    (out : ∀ k : TileIndex [BD], active masked D k → ooff s H D k < bounds "O") :
    FP.Structural.TraceSafe A bounds (kernel masked H S D Q BD) s := by
  unfold FP.Structural.TraceSafe
  rw [decomposition, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨prologue_safe A s H S D Q BD bounds length, ?_⟩
  intro initial prefixRun
  obtain ⟨q, ran, invariant⟩ := prologue_run A s masked H S D Q BD
  rw [ran] at prefixRun
  cases prefixRun
  rw [Safety.traceSafeList_append]
  constructor
  · simp only [Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeOp_unfold, evalOp_unfold, invariant.stop, true_and]
    refine ⟨?_, ?_⟩
    · simpa only [Nat.cast_zero] using range_safe A s initial masked H S D Q BD 0 invariant bounds mid log
    · cases step A (.forRangeInt "block_seq_n" (.constInt 0)
        (.ref .int [] "block_n_size") (.constNat 1) (body masked H S D Q BD)) initial <;> trivial
  · intro last loopRun
    obtain ⟨final, finalRun, finalInv⟩ := loop_run A s masked H S D Q BD
    rw [run_append, ran] at finalRun
    simp only [Option.bind_some] at finalRun
    rw [finalRun] at loopRun
    cases loopRun
    exact epilogue_safe A s last masked H S D Q BD finalInv bounds out

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2.AccessSafety
