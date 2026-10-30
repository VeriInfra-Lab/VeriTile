import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.RawValues
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelMath

/-! Kernel-only mathematical correctness. Every theorem constructs canonical
source execution, every active output, complete memory and the exterior frame.
Raw states provide all initial input values. Real inputs are arbitrary. Historical host/autograd results remain in Proofs/HostCorrect. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Correctness
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

theorem forward_kernel_correct_raw (N B : Nat) (covered : N≤B) (eps : ℝ) (bias : Bool)
    (u : State ℝ) :
    ∃ v,
      FP.Structural.exec (engine KernelMath.algebra)
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) u=some v ∧
      v.mem=KernelRows.forwardHeap KernelMath.algebra N B eps bias u ∧
      (∀ j : Fin B,j.val<N → v.mem "y" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.outputSpec N (fun k => RawValues.value u "x" (u.pids 0*N+k))
          (RawValues.value u "w") (fun k => if bias then RawValues.value u "b" k else 0) eps j.val)) ∧
      v.mem "r" (u.pids 0)=Cell.mk .real (RealProof.invSpec N
        (fun k => RawValues.value u "x" (u.pids 0*N+k)) eps) ∧
      v.mem "mu" (u.pids 0)=Cell.mk .real (RealProof.meanSpec N
        (fun k => RawValues.value u "x" (u.pids 0*N+k))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm,hy,hr,hmu,hf⟩ := KernelOutputs.forward_complete KernelMath.algebra N B eps bias u
  simp only [RawValues.raw,Nat.mul_zero,Nat.zero_add] at hy hr hmu
  refine ⟨v,he,hm,?_,?_,?_,hf⟩
  · intro j hj
    rw [hy j hj]
    exact KernelMath.output_real N B covered eps bias
      (fun k => RawValues.value u "x" (u.pids 0*N+k)) (RawValues.value u "w") (RawValues.value u "b") j hj
  · rw [hr]
    unfold KernelReference.savedInverse
    rw [KernelMath.inverse_real N B covered eps (fun k => RawValues.value u "x" (u.pids 0*N+k))]
    rfl
  · rw [hmu]
    unfold KernelReference.savedMean
    rw [KernelMath.mean_real N B covered (fun k => RawValues.value u "x" (u.pids 0*N+k))]
    rfl

theorem backward_kernel_correct_raw (N B : Nat) (covered : N≤B)
    (u : State ℝ) :
    ∃ v,
      FP.Structural.exec (engine KernelMath.algebra)
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) u=some v ∧
      v.mem=KernelRows.backwardHeap KernelMath.algebra N B u ∧
      (∀ j : Fin B,j.val<N → v.mem "dx" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.gradientSpec N (fun k => RawValues.value u "x" (u.pids 0*N+k))
          (fun k => RawValues.value u "dy" (u.pids 0*N+k)) (RawValues.value u "w")
          (RawValues.value u "mu" (u.pids 0)) (RawValues.value u "r" (u.pids 0)) j.val)) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm,hy,hf⟩ := KernelOutputs.backward_complete KernelMath.algebra N B u
  simp only [RawValues.raw,Nat.mul_zero,Nat.zero_add] at hy
  refine ⟨v,he,hm,?_,hf⟩
  intro j hj
  rw [hy j hj]
  exact KernelMath.gradient_real N B covered (fun k => RawValues.value u "x" (u.pids 0*N+k))
    (fun k => RawValues.value u "dy" (u.pids 0*N+k)) (RawValues.value u "w")
    (RawValues.value u "mu" (u.pids 0)) (RawValues.value u "r" (u.pids 0)) j hj

theorem affine_values_real (rows N : Nat)
    (u : State ℝ) (j : Nat) :
    RawValues.affineValues KernelMath.algebra rows N u j=
      (RealProof.weightGradientSpec N rows (fun row col => RawValues.value u "x" (row*N+col))
        (fun row col => RawValues.value u "dy" (row*N+col))
        (fun row => ⟨RawValues.value u "mu" row,RawValues.value u "r" row⟩) j,
       RealProof.biasGradientSpec N rows (fun row col => RawValues.value u "dy" (row*N+col)) j) := by
  unfold RawValues.affineValues
  rw [KernelMath.affine_real]
  by_cases hj : j<N <;> simp [hj,RealProof.weightGradientSpec,RealProof.biasGradientSpec]

theorem affine_kernel_correct_raw (rows N wc bc : Nat) (bias : Bool)
    (u : State ℝ) :
    ∃ v,
      FP.Structural.exec (engine KernelMath.algebra) (KernelAffine.kernel rows N wc bc bias) u=some v ∧
      v.mem=KernelAffine.heap KernelMath.algebra rows N wc bc (u.pids 0) bias u ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → v.mem "dw" (u.pids 0*128+j.val)=Cell.mk .real
        (RealProof.weightGradientSpec N rows (fun row col => RawValues.value u "x" (row*N+col))
          (fun row col => RawValues.value u "dy" (row*N+col))
          (fun row => ⟨RawValues.value u "mu" row,RawValues.value u "r" row⟩) (u.pids 0*128+j.val))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → v.mem "db" (u.pids 0*128+j.val)=Cell.mk .real
        (RealProof.biasGradientSpec N rows (fun row col => RawValues.value u "dy" (row*N+col)) (u.pids 0*128+j.val))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hm,hw,hb,hf⟩ := KernelOutputs.affine_complete KernelMath.algebra rows N wc bc bias u
  refine ⟨v,he,hm,?_,?_,hf⟩
  · intro j hj
    rw [hw j hj,RawValues.carry,affine_values_real]
    rfl
  · intro h j hj
    rw [hb h j hj,RawValues.carry,affine_values_real]
    rfl

#axiomsClean forward_kernel_correct_raw
#axiomsClean backward_kernel_correct_raw
#axiomsClean affine_kernel_correct_raw
#specNonCircular RealProof.outputSpec avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular RealProof.gradientSpec avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular RealProof.weightGradientSpec avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular RealProof.biasGradientSpec avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Correctness
