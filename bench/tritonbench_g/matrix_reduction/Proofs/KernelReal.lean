import bench.tritonbench_g.matrix_reduction.Contract
import bench.tritonbench_g.matrix_reduction.Proofs.Maximum
import bench.tritonbench_g.matrix_reduction.Proofs.TypedShape

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelReal
open VeriTile.Triton TypedSource
noncomputable section

theorem projection (storage : Storage) (X Y : RegionName) (sm sn sy M N : Nat) :
    (kernel storage X Y sm sn sy M N).eraseDType.toAlgKernel.body =
      (load_reduce_kernel X Y sm sn sy M N).toAlgKernel.body := by
  have body_eq (a b : ComputeKernel) (h : a.surfaceBody = b.surfaceBody) :
      a.toAlgKernel.body = b.toAlgKernel.body := by
    cases a with
    | mk ai ao ab =>
      cases b with
      | mk bi bo bb =>
        change ab = bb at h
        subst bb
        simp only [ComputeKernel.toAlgKernel_mk]
        cases ComputeStmt.listToAlgorithm? ab <;> rfl
  rw [body_eq _ _ (source_shape storage X Y sm sn sy M N)]
  simp only [ComputeKernel.eraseDType, mathematical_supported,
    ComputeKernel.toAlgKernel_fromKernelBody, mathematical_projection]
  simp [LegacyBody.load_reduce_kernel, ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType]

theorem coordinates (storage : Storage) (M N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin M → Fin N → ℝ)
    (input : ∀ i j, s.readMem "x" (i.val*N+j.val) = xs i j) :
    ∃ t, exec (kernel storage "x" "out" N 1 1 M N).eraseDType.toAlgKernel s = some t ∧
      (∀ i, KernelMath.IsMaximum (xs i) (t.readMem "out" i.val)) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ M ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := matrix_reduction_region_run "x" "out" N 1 1 M N hN s
    (fun k => xs k.1 k.2.1) (by intro k; simpa [xOffset] using input k.1 k.2.1)
  refine ⟨t, ?_, ?_, ?_⟩
  · simpa only [exec, projection] using he
  · intro i
    rw [hv (i,⟨0,hN⟩,())]
    exact KernelMath.reduction_maximum M N hN xs i
  · intro r o outside
    apply hf r o
    rcases outside with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun i => by omega

#axiomsClean projection
#axiomsClean coordinates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelReal
