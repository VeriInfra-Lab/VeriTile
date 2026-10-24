import bench.repairs.tritonbench_g.matrix_vector_multip.Correct
import bench.repairs.tritonbench_g.matrix_vector_multip.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.RealResult
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.NativeResult
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelBinding

/-! Optional actual Tensor applications and Python-check correspondence.
The primary mathematical and native kernel specifications are unchanged. -/
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.TensorBridge
open VeriTile VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section

theorem python_checks_iff_preconditions (x y : VeriTile.Torch.ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := checkInputs_iff x y

/-- Python admission plus ordinary kernel-memory/program binding supplies the
uint64 access interpretation used by the mathematical kernel theorem. -/
theorem real_kernel_contract_sound {x y : VeriTile.Torch.TensorMetadata}
    {c : KernelConfig} {pid : Nat} {mem : KernelMemory (rows x) (columns x)}
    (h : KernelContract x y c pid mem) :
    checkInputs (.tensor x) (.tensor y) = .ok () ∧
      KernelIndex.CheckedCorrespondence x y c pid mem :=
  ⟨(checkInputs_iff _ _).mpr h.inputs, KernelIndex.checked_correspondence h⟩

/-- The identical checked-domain/uint64 memory obligations accompany FP
execution; numerical inputs remain unrestricted FP32 words. -/
theorem fp32_kernel_contract_sound {x y : VeriTile.Torch.TensorMetadata}
    {c : KernelConfig} {pid : Nat} {mem : KernelMemory (rows x) (columns x)}
    (h : KernelContract x y c pid mem) :
    checkInputs (.tensor x) (.tensor y) = .ok () ∧
      KernelIndex.CheckedCorrespondence x y c pid mem :=
  ⟨(checkInputs_iff _ _).mpr h.inputs, KernelIndex.checked_correspondence h⟩

/-- Public Tensor-to-kernel result. The input State, aligned bases, bounds and
logical input values are all derived from the checked actual tensors through
the shared Tensor laws. No abstract matrixRead/vectorRead premise is required. -/
theorem mv_checked_kernel_correctness
    (x y : VeriTile.Torch.FP32.Tensor)
    (lx : VeriTile.Torch.FP32.TensorLaws x) (ly : VeriTile.Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata)
    (out : KernelBinding.Output x y lx ly h) (c : KernelConfig) (pid : Nat)
    (program : pid < (rows x.metadata + c.blockRows - 1) / c.blockRows)
    (programABI : pid < 2^31)
    (decode : VeriTile.Torch.FP32.Word → ℝ) (read : Nat → VeriTile.Torch.FP32.Word)
    (surrounding : State ℝ) :
    let N := rows x.metadata
    let M := columns x.metadata
    let s := KernelBinding.state x y lx ly h out decode read pid surrounding
    let matrix := TensorBinding.matrix x lx decode read
    let vector := TensorBinding.vector y ly decode read
    Spec.Real (
      checkInputs (.tensor x.metadata) (.tensor y.metadata) = .ok () ∧
      KernelIndex.CheckedCorrespondence x.metadata y.metadata c pid
        (KernelBinding.memory x y lx ly h out) ∧
      ∃ t, exec WholeMath.algebra (Kernels.source N M c.blockRows c.blockColumns) s = some t ∧
        t.mem = (reference N c.blockRows s
          (fun i => Correct.matrixVector M matrix vector (row c.blockRows pid i.val))).mem ∧
        (∀ i : Fin c.blockRows, row c.blockRows pid i.val < N →
          t.mem "C" (row c.blockRows pid i.val) = .mk .fp32
            (Correct.matrixVector M matrix vector (row c.blockRows pid i.val))) ∧
        (∀ r o, Outside N c.blockRows pid r o → t.mem r o = s.mem r o)) := by
  dsimp only
  refine ⟨(checkInputs_iff _ _).mpr h,
    KernelIndex.checked_correspondence (KernelBinding.contract x y lx ly h out c pid program programABI), ?_⟩
  have run := KernelRealResult.mv_kernel_correctness (rows x.metadata) (columns x.metadata)
    c.blockRows c.blockColumns (KernelIndex.config_bounds c).2.2.1
    (KernelBinding.state x y lx ly h out decode read pid surrounding)
    (TensorBinding.matrix x lx decode read) (TensorBinding.vector y ly decode read)
    (KernelBinding.matrix_read x y lx ly h out decode read pid surrounding)
    (KernelBinding.vector_read x y lx ly h out decode read pid surrounding)
  simpa only [KernelBinding.state_pid] using run

/-- Public FP32 Tensor-to-kernel equivalence. Shared Tensor laws discharge the
active InputWords premise from one physical memory map, preserving A/B aliases.
The output remains the independent ordered native graph on logical Tensor words. -/
theorem mv_checked_fp32_kernel_equiv
    (native : Scalars) (x y : VeriTile.Torch.FP32.Tensor)
    (lx : VeriTile.Torch.FP32.TensorLaws x) (ly : VeriTile.Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata)
    (out : KernelBinding.Output x y lx ly h) (c : KernelConfig)
    (g : Graph c.blockRows c.blockColumns) (pid : Nat)
    (program : pid < (rows x.metadata + c.blockRows - 1) / c.blockRows)
    (programABI : pid < 2^31) (read : Nat → VeriTile.Torch.FP32.Word)
    (surrounding : State Payload) :
    let N := rows x.metadata
    let M := columns x.metadata
    let s := KernelBinding.fpState x y lx ly h out read pid surrounding
    let matrix := TensorBinding.matrix x lx KernelBinding.bits read
    let vector := TensorBinding.vector y ly KernelBinding.bits read
    checkInputs (.tensor x.metadata) (.tensor y.metadata) = .ok () ∧
    KernelIndex.CheckedCorrespondence x.metadata y.metadata c pid
      (KernelBinding.memory x y lx ly h out) ∧
    ∃ t, exec (algebra native g) (Kernels.source N M c.blockRows c.blockColumns) s = some t ∧
      t.mem = (reference N c.blockRows s (fun i => Payload.word
        (KernelFP.formula native N M c.blockRows c.blockColumns g matrix vector pid i))).mem ∧
      (∀ i : Fin c.blockRows, row c.blockRows pid i.val < N →
        t.mem "C" (row c.blockRows pid i.val) = .mk .fp32 (.word
          (KernelFP.formula native N M c.blockRows c.blockColumns g matrix vector pid i))) ∧
      (∀ r o, Outside N c.blockRows pid r o → t.mem r o = s.mem r o) := by
  dsimp only
  refine ⟨(checkInputs_iff _ _).mpr h,
    KernelIndex.checked_correspondence (KernelBinding.contract x y lx ly h out c pid program programABI), ?_⟩
  have run := KernelNativeResult.mv_fp32_kernel_equiv native (rows x.metadata) (columns x.metadata)
    c.blockRows c.blockColumns g (KernelIndex.config_bounds c).2.2.1
    (KernelBinding.fpState x y lx ly h out read pid surrounding)
    (TensorBinding.matrix x lx KernelBinding.bits read)
    (TensorBinding.vector y ly KernelBinding.bits read)
    (KernelBinding.input_words x y lx ly h out read pid surrounding)
  simpa only [KernelBinding.fpState, KernelBinding.state_pid] using run

#axiomsClean python_checks_iff_preconditions
#axiomsClean real_kernel_contract_sound
#axiomsClean fp32_kernel_contract_sound
#axiomsClean mv_checked_kernel_correctness
#axiomsClean mv_checked_fp32_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.TensorBridge
