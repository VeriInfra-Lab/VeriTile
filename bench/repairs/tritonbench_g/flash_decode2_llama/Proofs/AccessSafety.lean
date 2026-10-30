import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafetyLoops

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
  simp [prologue, phi, ComputeKernel.surfaceBody,
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
    simp only [body,kernel,phi,llama,ComputeKernel.surfaceBody,List.getD,
      List.getElem?_cons_zero,List.getElem?_cons_succ] <;>
    simp [body, kernel, phi, llama, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
      numeric, numericLt, numericLe, natLt, natLe, bop, Region.cast, Option.bind,
      ofFloat, toFloat, ComputeDType.eraseDType,
      hb,hh,hl,hs,hv,he,hmax,hsum,ha,hm,hp]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  · constructor
    · intro k b
      simpa [voff, lane] using mid (k, b) (Or.inl rfl)
    · simpa [eoff] using log
  · constructor
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
    simp [epilogue, kernel, phi, llama, ComputeKernel.surfaceBody,
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
    Safety.rangeIntSafe A bounds "block_seq_n" (start : Int) (count s Q : Int) 1
      (body masked H S D Q BD) q := by
  apply Safety.rangeIntSafe_of_invariant A bounds "block_seq_n" (count s Q : Int) 1
    (body masked H S D Q BD)
    (fun i current => 0 ≤ i ∧ Inv A s masked H S D Q BD i.toNat current)
  · intro i inside current inv
    have cast : (i.toNat : Int) = i := Int.toNat_of_nonneg inv.1
    have bound : i.toNat < count s Q := by omega
    simpa only [cast] using
      body_safe A s current masked H S D Q BD i.toNat inv.2 bounds
        (mid i.toNat bound) (log i.toNat bound)
  · intro i inside current inv next executed
    have cast : (i.toNat : Int) = i := Int.toNat_of_nonneg inv.1
    obtain ⟨r,ran,hr⟩ := body_run A s current masked H S D Q BD i.toNat inv.2
    simp only [cast] at ran
    have eq := Option.some.inj (ran.symm.trans executed)
    subst next
    refine ⟨by omega, ?_⟩
    have succ : (i + 1).toNat = i.toNat + 1 := by omega
    simpa only [Int.natCast_one,succ] using hr
  · exact ⟨Int.natCast_nonneg _,by simpa only [Int.toNat_natCast] using I⟩

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
    refine ⟨range_safe A s initial masked H S D Q BD 0 invariant bounds mid log, ?_⟩
    cases step A (.forRangeInt "block_seq_n" (.constInt 0)
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
