import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Source
import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelInterface
import VeriTile.Triton.Float.StructuralSafety

/-! Safety of the actual Fast RoPE source, with its own proved prelude. The dynamic
four-head loop follows the actual interpreter states and the existing source
context invariant. Both loaded halves and both in-place stores are checked. -/
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem body_safe {A : Type} [Inhabited A] (M : Algebra A) (p : Source.Parameters)
    (backward : Bool) (row head : Nat) (cs ss : Source.Lane p → A) (s : State A)
    (h : Source.Context p row cs ss s) (bounds : RegionName → Nat)
    (access : ∀ side (i : Source.Lane p), Source.active p i →
      Source.offset p row head side i < bounds p.input) :
    Safety.traceSafeList M bounds (Source.body p backward)
      (s.setReg "k" .nat [] (fun _ => head)) := by
  cases backward
  all_goals simp (maxSteps := 1600000) [Source.body, Source.source,
    Kernels.rope, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    numericEq, Region.cast, State.setReg, h.row, h.columns, h.half, h.mask, h.cosine, h.sine,
    Source.active]
  all_goals refine ⟨?_, ?_, ?_, ?_⟩
  all_goals intro i active
  all_goals first
    | simpa [Source.offset] using access Bool.false (i, PUnit.unit) active
    | simpa [Source.offset] using access Bool.true (i, PUnit.unit) active

theorem loop_safe {A : Type} [Inhabited A] (M : Algebra A) (p : Source.Parameters)
    (backward : Bool) (row : Nat) (cs ss : Source.Lane p → A)
    (start count : Nat) (s : State A) (h : Source.Context p row cs ss s)
    (bounds : RegionName → Nat)
    (access : ∀ head, start ≤ head → head < start + count →
      ∀ side (i : Source.Lane p), Source.active p i →
        Source.offset p row head side i < bounds p.input) :
    Safety.rangeSafe M bounds "k" start (start + count) 1 (Source.body p backward) s := by
  induction count generalizing start s with
  | zero => simp [Safety.rangeSafe]
  | succ count ih =>
    let v := (Source.oneHead M p backward row start cs ss s h).val
    have ran : run M (Source.body p backward)
        (s.setReg "k" .nat [] (fun _ => start)) = some v :=
      (congrArg (fun body => run M body (s.setReg "k" .nat [] (fun _ => start)))
        (Source.body_shared p backward)).trans
          (Source.oneHead M p backward row start cs ss s h).property
    have context := (Source.oneHead_facts M p backward row start cs ss s h).1
    rw [Safety.rangeSafe]
    simp only [show (1 : Nat) ≠ 0 by decide, ↓reduceIte, show start < start + (count + 1) by omega]
    refine ⟨body_safe M p backward row start cs ss s h bounds
      (access start (by omega) (by omega)), ?_⟩
    rw [ran]
    have next := ih (start + 1) v context (fun head hlo hhi => access head (by omega) (by omega))
    simpa only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using next

theorem prelude_safe {A : Type} [Inhabited A] (M : Algebra A) (p : Source.Parameters)
    (backward : Bool) (s : State A) (bounds : RegionName → Nat)
    (sine : ∀ i : Source.Lane p, Source.active p i →
      s.pids 0 % p.length * p.sineStride + i.1.val < bounds p.sine)
    (cosine : ∀ i : Source.Lane p, Source.active p i →
      s.pids 0 % p.length * p.cosineStride + i.1.val < bounds p.cosine) :
    Safety.traceSafeList M bounds (Source.prelude p backward) s := by
  cases backward
  all_goals simp (maxSteps := 1600000) [Source.prelude, Source.source,
    Kernels.rope, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    numericEq, Region.cast, State.setReg, integralFloorDiv, integralMod]
  all_goals exact ⟨fun i h => sine (i, PUnit.unit) h, fun i h => cosine (i, PUnit.unit) h⟩

theorem source_safe {A : Type} [Inhabited A] (M : Algebra A) (p : Source.Parameters)
    (backward : Bool) (s : State A) (bounds : RegionName → Nat)
    (start : s.pids 1 * 4 ≤ p.heads)
    (sine : ∀ i : Source.Lane p, Source.active p i →
      s.pids 0 % p.length * p.sineStride + i.1.val < bounds p.sine)
    (cosine : ∀ i : Source.Lane p, Source.active p i →
      s.pids 0 % p.length * p.cosineStride + i.1.val < bounds p.cosine)
    (input : ∀ head, s.pids 1 * 4 ≤ head → head < min (s.pids 1 * 4 + 4) p.heads →
      ∀ side (i : Source.Lane p), Source.active p i →
        Source.offset p (s.pids 0) head side i < bounds p.input) :
    FP.Structural.TraceSafe M bounds (Source.source p backward) s := by
  unfold FP.Structural.TraceSafe
  rw [Source.decomposition, Safety.traceSafeList_append]
  refine ⟨prelude_safe M p backward s bounds sine cosine, ?_⟩
  intro t ran
  have same := Option.some.inj ((Source.prepared M p backward s).property.symm.trans ran)
  subst t
  have facts := Source.prepared_facts M p backward s
  rw [Safety.traceSafeList_cons]
  refine ⟨?_, fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe, Safety.safeOp.eq_def, evalOp_unfold, facts.2.2.1, facts.2.2.2,
    and_self, true_and]
  have endBound : s.pids 1 * 4 ≤ min (s.pids 1 * 4 + 4) p.heads := by omega
  have h := loop_safe M p backward (s.pids 0) _ _ (s.pids 1 * 4)
    (min (s.pids 1 * 4 + 4) p.heads - s.pids 1 * 4)
    (Source.prepared M p backward s).val facts.1 bounds
  have endEq : s.pids 1 * 4 + (min (s.pids 1 * 4 + 4) p.heads - s.pids 1 * 4) =
      min (s.pids 1 * 4 + 4) p.heads := by omega
  rw [endEq] at h
  exact h input

#axiomsClean body_safe
#axiomsClean loop_safe
#axiomsClean prelude_safe
#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.SourceSafety
