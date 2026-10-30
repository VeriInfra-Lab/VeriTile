import bench.tritonbench_g.matrix_vector_multip.Proofs.KernelSource
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.AccessSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip Typed WholeSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem prefix_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList A bounds (codePrefix p) s := by
  simp [codePrefix, WholeSource.source, Typed.kernel, Typed.source,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    step, evalExpr, evalComputeOp, evalOp_unfold, numeric, numericLt, natLt,
    bop, Region.cast, Option.bind]

theorem body_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α p)
    (s : State α) (ctx : Context p pid off memory acc s) (bounds : RegionName → Nat)
    (matrix : ∀ i j, i < p.rows → j < p.columns →
      i * p.strideRow + j * p.strideColumn < bounds p.a)
    (vector : ∀ j, j < p.columns → j * p.strideVector < bounds p.b) :
    Safety.traceSafeList A bounds (body p) (s.setReg "m" .nat [] (fun _ => off)) := by
  rcases ctx with ⟨hn, hm, hmask, ha, hb, hacc, hmem⟩
  simp [body, WholeSource.source, Typed.kernel, Typed.source,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericLt, natLt, bop, Region.cast, Option.bind, store,
    hn, hm, hmask, ha, hb, hacc, hmem, ofFloat, toFloat, ComputeDType.eraseDType]
  constructor
  · intro i j hi hj
    simpa [aOffset, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
      matrix (WholeSource.row p pid i.val) (off + j.val) hi hj
  · intro j hj
    simpa [bOffset, Nat.add_mul, Nat.add_comm] using vector (off + j.val) hj

theorem suffix_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α p)
    (s : State α) (ctx : Context p pid off memory acc s) (bounds : RegionName → Nat)
    (output : ∀ i : Fin p.blockRows, WholeSource.row p pid i.val < p.rows →
      WholeSource.row p pid i.val * p.strideOut < bounds p.out) :
    Safety.traceSafeList A bounds (suffix p) s := by
  rcases ctx with ⟨hn, hm, hmask, ha, hb, hacc, hmem⟩
  simp [suffix, WholeSource.source, Typed.kernel, Typed.source,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericLt, natLt, bop, Region.cast, Option.bind, store,
    hn, hm, hmask, ha, hb, hacc, hmem, ofFloat, toFloat, ComputeDType.eraseDType]
  exact output

theorem range_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (width : 0 < p.blockColumns) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α p)
    (s : State α) (ctx : Context p pid off memory acc s) (bounds : RegionName → Nat)
    (matrix : ∀ i j, i < p.rows → j < p.columns →
      i * p.strideRow + j * p.strideColumn < bounds p.a)
    (vector : ∀ j, j < p.columns → j * p.strideVector < bounds p.b) :
    Safety.rangeSafe A bounds "m" off p.columns p.blockColumns (body p) s := by
  rw [Safety.rangeSafe]
  simp only [Nat.ne_of_gt width, if_false]
  split
  · exact ⟨body_safe A p pid off memory acc s ctx bounds matrix vector, by
      rw [body_runs A p pid off memory acc s ctx]
      exact range_safe A p width pid (off + p.blockColumns) memory
        (nextAcc A p pid off memory acc) _ (body_context A p pid off memory acc s ctx)
        bounds matrix vector⟩
  · trivial
termination_by p.columns - off

/-- Every actual source access throughout the complete column loop is backed.
The proof follows the source's changing pointer registers and predecessor states. -/
theorem source_safe {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (width : 0 < p.blockColumns) (bounds : RegionName → Nat) (s : State α)
    (matrix : ∀ i j, i < p.rows → j < p.columns →
      i * p.strideRow + j * p.strideColumn < bounds p.a)
    (vector : ∀ j, j < p.columns → j * p.strideVector < bounds p.b)
    (output : ∀ i : Fin p.blockRows, WholeSource.row p (s.pids 0) i.val < p.rows →
      WholeSource.row p (s.pids 0) i.val * p.strideOut < bounds p.out) :
    FP.Structural.TraceSafe A bounds (WholeSource.source p) s := by
  unfold FP.Structural.TraceSafe
  rw [complete_source, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨prefix_safe A p bounds s, ?_⟩
  intro middle hprefix
  rw [prefix_runs] at hprefix
  cases hprefix
  rw [Safety.traceSafeList_append]
  constructor
  · simp only [Safety.traceSafeList_unfold, Safety.stepSafe_unfold]
    refine ⟨range_safe A p width (s.pids 0) 0 s.mem _ _
      (prefix_context A p s) bounds matrix vector, ?_⟩
    cases step A (.forRange "m" 0 p.columns p.blockColumns (body p))
      (prefixState A p (s.pids 0) s) <;> trivial
  · intro after executed
    obtain ⟨last, final, ran, ctx⟩ := range_runs A p width (s.pids 0) 0 s.mem
      (fun _ => A.literal (some .fp32) .real 0) _ (prefix_context A p s)
    simp only [run, step, ran, Option.bind_some] at executed
    cases executed
    exact suffix_safe A p (s.pids 0) last s.mem _ _ ctx bounds output

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.AccessSafety
