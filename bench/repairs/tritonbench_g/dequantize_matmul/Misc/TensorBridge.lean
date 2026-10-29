import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelAddresses

/-! Complete admitted Tensor applications and mixed-dtype pointer correspondence. -/
namespace VeriTile.Bench.Repairs.DequantizeMatmul.TensorBridge
open VeriTile.Triton FP.Structural KernelSource KernelModels KernelTensor
noncomputable section

/-- Complete mathematical dequantization kernel: the resolved signed INT8
input is promoted to Real and multiplied by its column scale. Both actual
quotations execute successfully, cover every output and preserve exterior
memory. Real interprets the FP32 source mathematically, without claiming
native FP32 arithmetic equals Real. torch.mm is a separate library consumer. -/
theorem kernel_correct (a : Direct.TensorInfo) (b : Int8Tensor) (scale : Tensor)
    (out : Option Direct.TensorInfo) (admitted : Direct.InputContract a (int8Info b) (fp32Info scale) out)
    (config : LaunchConfig) (member : config ∈ launchConfigs)
    (rest : Algebra ℝ) (read8 : Nat → BitVec 8) (scales : Nat → ℝ) (initial : State ℝ)
    (reads : Reads (parameters a b config) (int8Value read8 b) scales initial) :
    Spec.Real (∃ final,
      KernelGrid.run (realAlgebra rest) (parameters a b config) Bool.false
        (KernelGrid.order (parameters a b config)) initial = some final ∧
      KernelGrid.run (realAlgebra rest) (parameters a b config) Bool.true
        (KernelGrid.order (parameters a b config)) initial = some final ∧
      Result (parameters a b config)
        (fun i => KernelModels.mathematical (int8Value read8 b i)
          (scales (i % Direct.columns (int8Info b)))) initial final) ∧
    KernelAddresses.Exact (parameters a b config) := by
  refine ⟨?_, KernelAddresses.exact a b scale out admitted config member⟩
  simpa only [Spec.Real] using KernelRealResults.kernel_correct rest (parameters a b config)
    (launchConfigs_positive config member).1 (launchConfigs_positive config member).2
    (KernelTensor.separate a b config) (int8Value read8 b) scales initial reads

/-- Kernel-only native result from actual checked Tensor objects. The input
is signed INT8, scales/output retain all FP32 words, and the independent
formula is one source-ordered multiply after signed promotion. The candidate
only commutes the integer N-mask product. No finite guard, floating rewrite,
host matrix-product proof or expected-output premise is present. -/
theorem kernel_equiv (a : Direct.TensorInfo) (b : Int8Tensor) (scale output : Tensor)
    (out : Option Direct.TensorInfo) (admitted : Direct.InputContract a (int8Info b) (fp32Info scale) out)
    (lb : Int8Laws b) (ls : TensorLaws scale) (lo : TensorLaws output)
    (config : LaunchConfig) (member : config ∈ launchConfigs)
    (multiply : Word → Word → Word) (rest : Algebra Word)
    (read8 : Nat → BitVec 8) (read32 : Nat → Word) (initial : State Word)
    (binding : Binding a b scale output read8 read32 initial) :
    ∃ final,
      KernelGrid.run (nativeAlgebra multiply rest) (parameters a b config) Bool.false
        (KernelGrid.order (parameters a b config)) initial = some final ∧
      KernelGrid.run (nativeAlgebra multiply rest) (parameters a b config) Bool.true
        (KernelGrid.order (parameters a b config)) initial = some final ∧
      Result (parameters a b config)
        (fun i => nativeWord multiply (int8Value read8 b i)
          (scaleValue read32 scale ls (i % Direct.columns (int8Info b)))) initial final ∧
      (∀ i, i < Direct.columns a * Direct.columns (int8Info b) →
        outputStore output final i =
          ((output.dataPtr + BitVec.ofNat 64 (4*i)).toNat,
            nativeWord multiply (int8Value read8 b i)
              (scaleValue read32 scale ls (i % Direct.columns (int8Info b))))) ∧
      KernelAddresses.Exact (parameters a b config) := by
  obtain ⟨final, original, candidate, result⟩ := KernelNativeResults.kernel_equiv multiply rest
    (parameters a b config) (launchConfigs_positive config member).1
    (launchConfigs_positive config member).2 (KernelTensor.separate a b config)
    (int8Value read8 b) (scaleValue read32 scale ls) initial
    (KernelTensor.reads a b scale output out admitted lb ls config read8 read32 initial binding)
  refine ⟨final, original, candidate, result, ?_,
    KernelAddresses.exact a b scale out admitted config member⟩
  intro i hi
  have bound : i < output.metadata.numel := by
    rw [output_numel a b scale output read8 read32 initial binding]
    exact hi
  rw [fp32_pointer output lo binding.outputContiguous i bound]
  have value : final.mem "fpb" i = .mk .real
      (nativeWord multiply (int8Value read8 b i)
        (scaleValue read32 scale ls (i % Direct.columns (int8Info b)))) := result.1 i hi
  simp [outputStore, value]

alias python_checks_iff_preconditions := Direct.checkInputs_iff
alias native_input_pointer := KernelTensor.input_pointer
alias native_fp32_pointer := KernelTensor.fp32_pointer

#axiomsClean kernel_correct
#axiomsClean kernel_equiv
#axiomsClean python_checks_iff_preconditions
#axiomsClean native_input_pointer
#axiomsClean native_fp32_pointer
end
end VeriTile.Bench.Repairs.DequantizeMatmul.TensorBridge
