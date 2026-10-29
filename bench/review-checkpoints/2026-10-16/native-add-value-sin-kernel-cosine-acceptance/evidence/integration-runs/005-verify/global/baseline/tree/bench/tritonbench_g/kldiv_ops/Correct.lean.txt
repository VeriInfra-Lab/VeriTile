import bench.tritonbench_g.kldiv_ops.Proofs.SourceExecution
import bench.tritonbench_g.kldiv_ops.Proofs.Contracts
import bench.tritonbench_g.kldiv_ops.Proofs.Precision
import bench.tritonbench_g.kldiv_ops.Proofs.RealWitness
import bench.tritonbench_g.kldiv_ops.Proofs.LegacyReal

/-! Whole-row ideal-real contracts for the complete typed source and candidate.
The statement pins each source stride, active bounds, actual successful flat
execution, the independent mathematical loss or gradient, and every unwritten
raw memory cell. Python wrapper reductions and grad_output scaling are separate. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps
open VeriTile.Triton TypedSource
open scoped VeriTile.Triton.MaskedKernelIO₂
open scoped VeriTile.Triton.MaskedKernelIO₁
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Compute-facing correctness for the default backward one-block slice. -/
specification kldiv_backward_default_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_default]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_default_correct target_ptr new_grads_ptr
    target_stride new_grads_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the log-target backward one-block slice. -/
specification kldiv_backward_log_target_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_backward_log_target]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_backward_log_target_correct target_ptr new_grads_ptr
    target_stride new_grads_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` one-block slice. -/
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_forward_default_none]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_forward_default_none_correct y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps s s' hOutInj hExec i
  simpa [hActive] using h

/-- Compute-facing correctness for the forward log-target one-block slice. -/
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [kldiv_forward_log_target_none]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := kldiv_forward_log_target_none_correct y_ptr gt_ptr loss_ptr
    y_stride gt_stride loss_stride n_cols BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- **The headline on the IO surface**: the `log_target = True`,
`reduction = None` forward slice implements `exp(y_true) · (y_true − y)` on its
masked IO signature — for every disjoint flat placement of the three buffers,
every program id whose active lanes are in bounds, and every launch state whose
input windows hold `ys`/`gts` at the active lanes, the translated pointer kernel
terminates, every active output lane holds the spec value, and every other
memory cell is unchanged.

This is the audit-once surface; the `Realizes_without_Rounding` face above
states the same computation against a declared write map but does not pin the
flat placement. -/
specification kldiv_forward_log_target_none_correctness
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    kldivForwardLogTargetIO y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE
      ⊨ fun ys gts i => Real.exp (gts i) * (gts i - ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact kldiv_forward_log_target_none_flattenOk y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact kldiv_forward_log_target_none_traceSafe y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE bounds s h1 h2 h3
  · intro s₀ ys gts hy hgt
    obtain ⟨s1, hexec, hval, hframe⟩ := kldiv_forward_log_target_none_region_run
      y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols BLOCK_SIZE
      s₀ ys gts hy hgt
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- **The headline on the IO surface**: the `log_target = False` (default),
`reduction = None` forward slice implements `y_true · (log(max(y_true, eps)) − y)` on its
masked IO signature — for every disjoint flat placement of the three buffers,
every program id whose active lanes are in bounds, and every launch state whose
input windows hold `ys`/`gts` at the active lanes, the translated pointer kernel
terminates, every active output lane holds the spec value, and every other
memory cell is unchanged.

This is the audit-once surface; the `Realizes_without_Rounding` face above
states the same computation against a declared write map but does not pin the
flat placement. -/
specification kldiv_forward_default_none_correctness
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) :
    kldivForwardDefaultIO y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps
      ⊨ fun ys gts i => gts i * (Real.log (max (gts i) eps) - ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact kldiv_forward_default_none_flattenOk y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps
  · intro bounds s h1 h2 h3 _
    exact kldiv_forward_default_none_traceSafe y_ptr gt_ptr loss_ptr
      y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps bounds s h1 h2 h3
  · intro s₀ ys gts hy hgt
    obtain ⟨s1, hexec, hval, hframe⟩ := kldiv_forward_default_none_region_run
      y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps
      s₀ ys gts hy hgt
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- **The headline on the IO surface** for the `log_target = False` backward
slice: the gradient is `−target` on every active lane. -/
specification kldiv_backward_default_correctness
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    kldivBackwardDefaultIO target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE
      ⊨ fun ts i => ts i * (0.0 - 1) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact kldiv_backward_default_flattenOk target_ptr new_grads_ptr
      target_stride new_grads_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact kldiv_backward_default_traceSafe target_ptr new_grads_ptr
      target_stride new_grads_stride n_cols BLOCK_SIZE bounds s h1 h2
  · intro s₀ ts hin
    obtain ⟨s1, hexec, hval, hframe⟩ := kldiv_backward_default_region_run
      target_ptr new_grads_ptr target_stride new_grads_stride n_cols BLOCK_SIZE
      s₀ ts hin
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- **The headline on the IO surface** for the `log_target = True` backward
slice: the gradient is `−exp(target)` on every active lane. -/
specification kldiv_backward_log_target_correctness
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    kldivBackwardLogTargetIO target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE
      ⊨ fun ts i => 0.0 - Real.exp (ts i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact kldiv_backward_log_target_flattenOk target_ptr new_grads_ptr
      target_stride new_grads_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact kldiv_backward_log_target_traceSafe target_ptr new_grads_ptr
      target_stride new_grads_stride n_cols BLOCK_SIZE bounds s h1 h2
  · intro s₀ ts hin
    obtain ⟨s1, hexec, hval, hframe⟩ := kldiv_backward_log_target_region_run
      target_ptr new_grads_ptr target_stride new_grads_stride n_cols BLOCK_SIZE
      s₀ ts hin
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
specification forward_default_element_correct (pred target output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ)
    (_hSource : ForwardDefaultSupported pred target output 0)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardElementIO
      (forwardDefault pred target output v Y T O sy st so N B eps 0)
      A s Y T O sy st so N eps Bool.false) := by
  unfold Contracts.ForwardElementIO
  refine ⟨Source.forward_default_safe A.extent s pred target output v Y T O sy st so N B eps 0 hB
    (fun _ ↦ ⟨hYO, hTO⟩) hby hbt (fun _ ↦ hbo) (by simp), ?_⟩
  obtain ⟨t, he, hv, hf⟩ := FlatIO.element_run A s Y T O sy st so N B eps Bool.false hd hO hc hu hB hYO hTO hby hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  exact (Source.forward_default_flat_exec A pred target output v
    Y T O sy st so N B eps 0 (A.flattenState s)).trans he

/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
specification forward_log_element_correct (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ)
    (_hSource : ForwardLogSupported pred target output 0)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardElementIO
      (forwardLog pred target output v Y T O sy st so N B eps 0)
      A s Y T O sy st so N eps Bool.true) := by
  unfold Contracts.ForwardElementIO
  refine ⟨Source.forward_log_safe A.extent s pred target output v Y T O sy st so N B eps 0 hB
    (fun _ ↦ ⟨hYO, hTO⟩) hby hbt (fun _ ↦ hbo) (by simp), ?_⟩
  obtain ⟨t, he, hv, hf⟩ := FlatIO.element_run A s Y T O sy st so N B eps Bool.true hd hO hc hu hB hYO hTO hby hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  exact (Source.forward_log_flat_exec A pred target output v
    Y T O sy st so N B eps 0 (A.flattenState s)).trans he

/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
specification forward_default_reduction_correct (pred target output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (_hSource : ForwardDefaultSupported pred target output reduction)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hR : reduction ≠ 0)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : s.pids 0 * so < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardReductionIO
      (forwardDefault pred target output v Y T O sy st so N B eps reduction)
      A s Y T O sy st so N eps Bool.false) := by
  unfold Contracts.ForwardReductionIO
  refine ⟨Source.forward_default_safe A.extent s pred target output v Y T O sy st so N B eps reduction hB
    (fun hr ↦ False.elim (hR hr)) hby hbt (fun hr ↦ False.elim (hR hr)) (fun _ ↦ hbo), ?_⟩
  obtain ⟨t, he, hv, hf⟩ := FlatIO.reduction_run A s Y T O sy st so N B eps Bool.false reduction hd hO hc hu hB hR hby hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  exact (Source.forward_default_flat_exec A pred target output v
    Y T O sy st so N B eps reduction (A.flattenState s)).trans he

/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
specification forward_log_reduction_correct (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (_hSource : ForwardLogSupported pred target output reduction)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hR : reduction ≠ 0)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : s.pids 0 * so < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardReductionIO
      (forwardLog pred target output v Y T O sy st so N B eps reduction)
      A s Y T O sy st so N eps Bool.true) := by
  unfold Contracts.ForwardReductionIO
  refine ⟨Source.forward_log_safe A.extent s pred target output v Y T O sy st so N B eps reduction hB
    (fun hr ↦ False.elim (hR hr)) hby hbt (fun hr ↦ False.elim (hR hr)) (fun _ ↦ hbo), ?_⟩
  obtain ⟨t, he, hv, hf⟩ := FlatIO.reduction_run A s Y T O sy st so N B eps Bool.true reduction hd hO hc hu hB hR hby hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  exact (Source.forward_log_flat_exec A pred target output v
    Y T O sy st so N B eps reduction (A.flattenState s)).trans he

/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
specification backward_default_correct (target output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (T O : RegionName)
    (st so N B : Nat)
    (_hSource : BackwardSourceSupported target output Bool.false)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.BackwardIO
      (backwardDefault target output v T O st so N B)
      A s T O st so N Bool.false) := by
  unfold Contracts.BackwardIO
  refine ⟨Source.backward_default_safe A.extent s target output v T O st so N B hTO hbt hbo, ?_⟩
  obtain ⟨t, he, hv, hf⟩ := FlatIO.backward_run A s T O st so N B Bool.false hd hO hc hu hB hTO hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  exact (Source.backward_default_flat_exec A target output v
    T O st so N B (A.flattenState s)).trans he

/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
specification backward_log_correct (target : MathStorage) (output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (T O : RegionName)
    (st so N B : Nat)
    (_hSource : BackwardSourceSupported target.storage output Bool.true)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.BackwardIO
      (backwardLog target output v T O st so N B)
      A s T O st so N Bool.true) := by
  unfold Contracts.BackwardIO
  refine ⟨Source.backward_log_safe A.extent s target output v T O st so N B hTO hbt hbo, ?_⟩
  obtain ⟨t, he, hv, hf⟩ := FlatIO.backward_run A s T O st so N B Bool.true hd hO hc hu hB hTO hbt hbo
  refine ⟨t, ?_, hv, hf⟩
  exact (Source.backward_log_flat_exec A target output v
    T O st so N B (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps
