import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelBound
import bench.repairs.tritonbench_g.fast_layernorm.Correct
import bench.repairs.tritonbench_g.fast_layernorm.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Settings
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage

/-! Optional checked Tensor applications and storage interpretations of the
forward, backward and affine kernel results. The primary entries retain each
source-stage mathematical/native theorem and its full memory frame. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.TensorBridge
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelInputs KernelNative
noncomputable section

theorem python_checks_iff_preconditions (x w : VeriTile.Torch.ArgumentMetadata)
    (bias : Option VeriTile.Torch.ArgumentMetadata) (eps : Contract.Epsilon) :
    Contract.checkArguments x w bias eps=.ok () ↔ Contract.AdmittedArguments x w bias eps :=
  Contract.checkArguments_iff x w bias eps
theorem python_backward_checks_iff_preconditions (dy x : VeriTile.Torch.TensorMetadata) :
    Contract.checkGradient dy x=.ok () ↔ Contract.GradientContract dy x :=
  Contract.checkGradient_iff dy x

abbrev checked_forward_views := @KernelInputs.forwardViews
abbrev source_gradient_normalization := @KernelInputs.normalized

theorem source_access_pointer {α : Type} [Inhabited α] {read : Nat → α}
    (v : ReadView α read) (i : Nat) (active : i<v.tensor.metadata.numel) :
    (BitVec.ofNat 64 ((VeriTile.Torch.FP32.dataView v.tensor v.laws (v.checked.dense rfl)).address i)).toNat=
      (VeriTile.Torch.FP32.dataView v.tensor v.laws (v.checked.dense rfl)).address i :=
  KernelInputs.native_address v i active

#axiomsClean python_checks_iff_preconditions
#axiomsClean python_backward_checks_iff_preconditions
#axiomsClean source_access_pointer

namespace Correctness
open VeriTile.Bench.Repairs.FastLayernorm.Correctness
variable {read : Nat → ℝ}

theorem forward_kernel_correct (N B : Nat) (covered : N≤B) (eps : ℝ) (bias : Bool)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ) :
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine KernelMath.algebra)
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap KernelMath.algebra N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "y" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.outputSpec N (fun k => value views u "x" (u.pids 0*N+k))
          (value views u "w") (fun k => if bias then value views u "b" k else 0) eps j.val)) ∧
      v.mem "r" (u.pids 0)=Cell.mk .real (RealProof.invSpec N
        (fun k => value views u "x" (u.pids 0*N+k)) eps) ∧
      v.mem "mu" (u.pids 0)=Cell.mk .real (RealProof.meanSpec N
        (fun k => value views u "x" (u.pids 0*N+k))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  obtain ⟨hp,v,he,hm,hy,hr,hmu,hf⟩ := KernelBound.forward KernelMath.algebra N B eps bias views u
  refine ⟨hp,v,he,hm,?_,?_,?_,hf⟩
  · intro j hj
    rw [hy j hj]
    exact KernelMath.output_real N B covered eps bias
      (fun k => value views u "x" (u.pids 0*N+k)) (value views u "w") (value views u "b") j hj
  · rw [hr]
    unfold KernelReference.savedInverse
    rw [KernelMath.inverse_real N B covered eps (fun k => value views u "x" (u.pids 0*N+k))]
    rfl
  · rw [hmu]
    unfold KernelReference.savedMean
    rw [KernelMath.mean_real N B covered (fun k => value views u "x" (u.pids 0*N+k))]
    rfl

theorem backward_kernel_correct (N B : Nat) (covered : N≤B)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ) :
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine KernelMath.algebra)
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) (physical views u)=some v ∧
      v.mem=KernelRows.backwardHeap KernelMath.algebra N B (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "dx" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.gradientSpec N (fun k => value views u "x" (u.pids 0*N+k))
          (fun k => value views u "dy" (u.pids 0*N+k)) (value views u "w")
          (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j.val)) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  obtain ⟨hp,v,he,hm,hy,hf⟩ := KernelBound.backward KernelMath.algebra N B views u
  refine ⟨hp,v,he,hm,?_,hf⟩
  intro j hj
  rw [hy j hj]
  exact KernelMath.gradient_real N B covered (fun k => value views u "x" (u.pids 0*N+k))
    (fun k => value views u "dy" (u.pids 0*N+k)) (value views u "w")
    (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j hj

theorem affine_values_real (rows N : Nat) (views : RegionName → Option (ReadView ℝ read))
    (u : State ℝ) (j : Nat) :
    KernelBound.affineValues KernelMath.algebra rows N views u j=
      (RealProof.weightGradientSpec N rows (fun row col => value views u "x" (row*N+col))
        (fun row col => value views u "dy" (row*N+col))
        (fun row => ⟨value views u "mu" row,value views u "r" row⟩) j,
       RealProof.biasGradientSpec N rows (fun row col => value views u "dy" (row*N+col)) j) := by
  unfold KernelBound.affineValues
  rw [KernelMath.affine_real]
  by_cases hj : j<N <;> simp [hj,RealProof.weightGradientSpec,RealProof.biasGradientSpec]

theorem affine_kernel_correct (rows N wc bc : Nat) (bias : Bool)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ) :
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine KernelMath.algebra) (KernelAffine.kernel rows N wc bc bias) (physical views u)=some v ∧
      v.mem=KernelAffine.heap KernelMath.algebra rows N wc bc (u.pids 0) bias (logical views u) ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → v.mem "dw" (u.pids 0*128+j.val)=Cell.mk .real
        (RealProof.weightGradientSpec N rows (fun row col => value views u "x" (row*N+col))
          (fun row col => value views u "dy" (row*N+col))
          (fun row => ⟨value views u "mu" row,value views u "r" row⟩) (u.pids 0*128+j.val))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → v.mem "db" (u.pids 0*128+j.val)=Cell.mk .real
        (RealProof.biasGradientSpec N rows (fun row col => value views u "dy" (row*N+col)) (u.pids 0*128+j.val))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=(physical views u).mem q o) := by
  obtain ⟨hp,v,he,hm,hw,hb,hf⟩ := KernelBound.affine KernelMath.algebra rows N wc bc bias views u
  refine ⟨hp,v,he,hm,?_,?_,hf⟩
  · intro j hj
    rw [hw j hj,affine_values_real]
    rfl
  · intro h j hj
    rw [hb h j hj,affine_values_real]
    rfl

#axiomsClean forward_kernel_correct
#axiomsClean backward_kernel_correct
#axiomsClean affine_kernel_correct

variable {read : Nat → ℝ}

/-- Checked wrapper specialization: every read view is derived from the
original Tensor, and the launch block is exactly next_power_of_2(N). -/
theorem checked_forward_kernel_correct (read : Nat → ℝ) (x w : VeriTile.Torch.FP32.Tensor)
    (b : Option VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x)
    (lw : VeriTile.Torch.FP32.TensorLaws w) (lb : ∀ t,b=some t → VeriTile.Torch.FP32.TensorLaws t)
    (eps : Contract.Epsilon)
    (h : Contract.InputContract x.metadata w.metadata (b.map VeriTile.Torch.FP32.Tensor.metadata) eps)
    (u : State ℝ) :
    let N := Contract.width x.metadata
    let B := N.nextPowerOfTwo
    let views := forwardViews read x w b lx lw lb eps h
    let bias := b.isSome
    let eps := eps.value
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine KernelMath.algebra)
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap KernelMath.algebra N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "y" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.outputSpec N (fun k => value views u "x" (u.pids 0*N+k))
          (value views u "w") (fun k => if bias then value views u "b" k else 0) eps j.val)) ∧
      v.mem "r" (u.pids 0)=Cell.mk .real (RealProof.invSpec N
        (fun k => value views u "x" (u.pids 0*N+k)) eps) ∧
      v.mem "mu" (u.pids 0)=Cell.mk .real (RealProof.meanSpec N
        (fun k => value views u "x" (u.pids 0*N+k))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) :=
  forward_kernel_correct (Contract.width x.metadata) (Contract.width x.metadata).nextPowerOfTwo
    (Settings.nextPower_covers _) eps.value b.isSome (forwardViews read x w b lx lw lb eps h) u


theorem forward_tensor_correct (N B : Nat) (covered : N≤B) (eps : ℝ) (bias : Bool)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ) (_access : (KernelInputs.ForwardReads views (u.pids 0) N bias))
    (outputs : KernelStorage.Placement (KernelOutputs.forwardFootprint N B (u.pids 0))) :
    KernelBound.Pointers views ∧ ∃ v,
      (FP.Structural.exec (engine KernelMath.algebra)
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap KernelMath.algebra N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "y" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.outputSpec N (fun k => value views u "x" (u.pids 0*N+k))
          (value views u "w") (fun k => if bias then value views u "b" k else 0) eps j.val)) ∧
      v.mem "r" (u.pids 0)=Cell.mk .real (RealProof.invSpec N
        (fun k => value views u "x" (u.pids 0*N+k)) eps) ∧
      v.mem "mu" (u.pids 0)=Cell.mk .real (RealProof.meanSpec N
        (fun k => value views u "x" (u.pids 0*N+k))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o)) ∧
      KernelStorage.TensorImage VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.realAPI
        outputs (fun _ _ => Cell.read .real) v read :=
  KernelStorage.attach (forward_kernel_correct N B covered eps bias views u)
    VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.realAPI outputs (fun _ _ => Cell.read .real) read

theorem backward_tensor_correct (N B : Nat) (covered : N≤B)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ) (_access : (KernelInputs.BackwardReads views (u.pids 0) N))
    (outputs : KernelStorage.Placement (KernelOutputs.backwardFootprint N B (u.pids 0))) :
    KernelBound.Pointers views ∧ ∃ v,
      (FP.Structural.exec (engine KernelMath.algebra)
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) (physical views u)=some v ∧
      v.mem=KernelRows.backwardHeap KernelMath.algebra N B (logical views u) ∧
      (∀ j : Fin B,j.val<N → v.mem "dx" (u.pids 0*N+j.val)=Cell.mk .real
        (RealProof.gradientSpec N (fun k => value views u "x" (u.pids 0*N+k))
          (fun k => value views u "dy" (u.pids 0*N+k)) (value views u "w")
          (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j.val)) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o)) ∧
      KernelStorage.TensorImage VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.realAPI
        outputs (fun _ _ => Cell.read .real) v read :=
  KernelStorage.attach (backward_kernel_correct N B covered views u)
    VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.realAPI outputs (fun _ _ => Cell.read .real) read

theorem affine_tensor_correct (rows N wc bc : Nat) (bias : Bool)
    (views : RegionName → Option (ReadView ℝ read)) (u : State ℝ) (_access : (KernelInputs.AffineReads views rows N (u.pids 0)))
    (outputs : KernelStorage.Placement (KernelOutputs.affineFootprint wc bc (u.pids 0) bias)) :
    KernelBound.Pointers views ∧ ∃ v,
      (FP.Structural.exec (engine KernelMath.algebra) (KernelAffine.kernel rows N wc bc bias) (physical views u)=some v ∧
      v.mem=KernelAffine.heap KernelMath.algebra rows N wc bc (u.pids 0) bias (logical views u) ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → v.mem "dw" (u.pids 0*128+j.val)=Cell.mk .real
        (RealProof.weightGradientSpec N rows (fun row col => value views u "x" (row*N+col))
          (fun row col => value views u "dy" (row*N+col))
          (fun row => ⟨value views u "mu" row,value views u "r" row⟩) (u.pids 0*128+j.val))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → v.mem "db" (u.pids 0*128+j.val)=Cell.mk .real
        (RealProof.biasGradientSpec N rows (fun row col => value views u "dy" (row*N+col)) (u.pids 0*128+j.val))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=(physical views u).mem q o)) ∧
      KernelStorage.TensorImage VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.realAPI
        outputs (fun _ _ => Cell.read .real) v read :=
  KernelStorage.attach (affine_kernel_correct rows N wc bc bias views u)
    VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.realAPI outputs (fun _ _ => Cell.read .real) read

#axiomsClean checked_forward_kernel_correct
#axiomsClean forward_tensor_correct
#axiomsClean backward_tensor_correct
#axiomsClean affine_tensor_correct
end Correctness

namespace Equivalence
open VeriTile.Bench.Repairs.FastLayernorm.Equivalence
variable {read : Nat → Term}
theorem forward_graph_equiv (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B : Nat) (eps : ℝ) (bias : Bool)
    (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R I (S "y" (u.pids 0*N+j.val)) N eps bias (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=observe I (S "r" (u.pids 0))
        (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  obtain ⟨hp,v,he,hm,hy,hr,hmu,hf⟩ := KernelBound.forward (algebra R) N B eps bias views u
  exact ⟨hp,v,he,hm,fun j hj => congrArg (observe I (S "y" (u.pids 0*N+j.val))) (hy j hj),
    congrArg (observe I (S "r" (u.pids 0))) hr,congrArg (observe I (S "mu" (u.pids 0))) hmu,hf⟩

/-- The independent reference receives the actual converted epsilon word.
The only additional obligation identifies that one scalar leaf; the proof
uses it to rewrite the native output and saved-inverse reference, including
coinciding literals, without changing the source graph or compiler schedule. -/
theorem forward_scalar_graph_equiv (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B : Nat) (eps : ℝ) (epsilon : Word) (bias : Bool)
    (conversion : literal32 I eps=epsilon)
    (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    (KernelBound.Pointers views ∧
      ∀ schedule,eval I schedule (.literal (some .fp32) .real eps)=epsilon) ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R (withEpsilon I eps epsilon) (S "y" (u.pids 0*N+j.val)) N eps bias
          (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=
        observe (withEpsilon I eps epsilon) (S "r" (u.pids 0))
          (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  rw [withEpsilon_eq I eps epsilon conversion]
  obtain ⟨hp,v,hv⟩ := forward_graph_equiv R I S N B eps bias views u
  exact ⟨⟨hp,fun schedule => (eval_literal32 I schedule eps).trans conversion⟩,v,hv⟩

theorem backward_graph_equiv (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B : Nat) (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) (physical views u)=some v ∧
      v.mem=KernelRows.backwardHeap (algebra R) N B (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "dx" (u.pids 0*N+j.val)) (v.mem "dx" (u.pids 0*N+j.val))=
        nativeGradientReference R I (S "dx" (u.pids 0*N+j.val)) N (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "dy" (u.pids 0*N+k.val)) (fun k => value views u "w" k.val)
          (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) := by
  obtain ⟨hp,v,he,hm,hy,hf⟩ := KernelBound.backward (algebra R) N B views u
  exact ⟨hp,v,he,hm,fun j hj => congrArg (observe I (S "dx" (u.pids 0*N+j.val))) (hy j hj),hf⟩

theorem affine_graph_equiv (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (rows N wc bc : Nat) (bias : Bool)
    (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine (algebra R)) (KernelAffine.kernel rows N wc bc bias) (physical views u)=some v ∧
      v.mem=KernelAffine.heap (algebra R) rows N wc bc (u.pids 0) bias (logical views u) ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → observe I (S "dw" (u.pids 0*128+j.val)) (v.mem "dw" (u.pids 0*128+j.val))=
        observe I (S "dw" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (KernelBound.affineValues (algebra R) rows N views u (u.pids 0*128+j.val)).1))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → observe I (S "db" (u.pids 0*128+j.val)) (v.mem "db" (u.pids 0*128+j.val))=
        observe I (S "db" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (KernelBound.affineValues (algebra R) rows N views u (u.pids 0*128+j.val)).2))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=(physical views u).mem q o) := by
  obtain ⟨hp,v,he,hm,hw,hb,hf⟩ := KernelBound.affine (algebra R) rows N wc bc bias views u
  exact ⟨hp,v,he,hm,fun j hj => congrArg (observe I (S "dw" (u.pids 0*128+j.val))) (hw j hj),
    fun h j hj => congrArg (observe I (S "db" (u.pids 0*128+j.val))) (hb h j hj),hf⟩

/-- Public native specialization: the graph, primitive meanings and scalar
conversion are bound to the same selected object and original host epsilon. -/
theorem forward_kernel_equiv (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule) (N B : Nat)
    (binding : Binding platform decoder N B .forward R S) (eps : ℝ) (bias : Bool)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (_abi : Compiler.Bound platform binding.selected .forward [N,N,N] hostEpsilon bias)
    (conversion : EpsilonConversion platform binding.selected.object P eps hostEpsilon)
    (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    let I := P binding.selected.object.primitives
    let epsilon := epsilonWord platform binding.selected.object hostEpsilon
    (KernelBound.Pointers views ∧
      ∀ schedule,eval I schedule (.literal (some .fp32) .real eps)=epsilon) ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R (withEpsilon I eps epsilon) (S "y" (u.pids 0*N+j.val)) N eps bias
          (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=
        observe (withEpsilon I eps epsilon) (S "r" (u.pids 0))
          (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) :=
  forward_scalar_graph_equiv R (P binding.selected.object.primitives) S N B eps
    (epsilonWord platform binding.selected.object hostEpsilon) bias conversion views u

theorem backward_kernel_equiv (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule) (N B : Nat)
    (binding : Binding platform decoder N B .backward R S)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (_abi : Compiler.Bound platform binding.selected .backward [N,N,N,N] hostEpsilon binding.selected.object.signature.hasBias)
    (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    let I := P binding.selected.object.primitives
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) (physical views u)=some v ∧
      v.mem=KernelRows.backwardHeap (algebra R) N B (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "dx" (u.pids 0*N+j.val)) (v.mem "dx" (u.pids 0*N+j.val))=
        nativeGradientReference R I (S "dx" (u.pids 0*N+j.val)) N (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "dy" (u.pids 0*N+k.val)) (fun k => value views u "w" k.val)
          (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) :=
  backward_graph_equiv R (P binding.selected.object.primitives) S N B views u

theorem affine_kernel_equiv (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule) (rows N B wc bc : Nat)
    (binding : Binding platform decoder N B .affineBackward R S) (bias : Bool)
    (_abi : Compiler.Bound platform binding.selected .affineBackward [rows,N,wc,bc] (.boolean Bool.false) bias)
    (views : RegionName → Option (ReadView Term read)) (u : State Term) :
    let I := P binding.selected.object.primitives
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine (algebra R)) (KernelAffine.kernel rows N wc bc bias) (physical views u)=some v ∧
      v.mem=KernelAffine.heap (algebra R) rows N wc bc (u.pids 0) bias (logical views u) ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → observe I (S "dw" (u.pids 0*128+j.val)) (v.mem "dw" (u.pids 0*128+j.val))=
        observe I (S "dw" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (KernelBound.affineValues (algebra R) rows N views u (u.pids 0*128+j.val)).1))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → observe I (S "db" (u.pids 0*128+j.val)) (v.mem "db" (u.pids 0*128+j.val))=
        observe I (S "db" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (KernelBound.affineValues (algebra R) rows N views u (u.pids 0*128+j.val)).2))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=(physical views u).mem q o) :=
  affine_graph_equiv R (P binding.selected.object.primitives) S rows N wc bc bias views u

#axiomsClean forward_kernel_equiv
#axiomsClean backward_kernel_equiv
#axiomsClean affine_kernel_equiv
#axiomsClean forward_graph_equiv
#axiomsClean forward_scalar_graph_equiv
#axiomsClean backward_graph_equiv
#axiomsClean affine_graph_equiv

variable {read : Nat → Term}

def physicalWords (read : Nat → VeriTile.Torch.FP32.Word) : Nat → Term :=
  fun address => .leaf "physical" address (read address).bits


/-- All FP32 payloads enter through the checked original Tensor pointers. -/
theorem checked_forward_graph_equiv (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (read : Nat → VeriTile.Torch.FP32.Word) (x w : VeriTile.Torch.FP32.Tensor)
    (b : Option VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x)
    (lw : VeriTile.Torch.FP32.TensorLaws w) (lb : ∀ t,b=some t → VeriTile.Torch.FP32.TensorLaws t)
    (eps : Contract.Epsilon)
    (h : Contract.InputContract x.metadata w.metadata (b.map VeriTile.Torch.FP32.Tensor.metadata) eps)
    (u : State Term) :
    let N := Contract.width x.metadata
    let B := N.nextPowerOfTwo
    let views := forwardViews (physicalWords read) x w b lx lw lb eps h
    let bias := b.isSome
    let eps := eps.value
    KernelBound.Pointers views ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R I (S "y" (u.pids 0*N+j.val)) N eps bias (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=observe I (S "r" (u.pids 0))
        (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) :=
  forward_graph_equiv R I S (Contract.width x.metadata) (Contract.width x.metadata).nextPowerOfTwo
    eps.value b.isSome (forwardViews (physicalWords read) x w b lx lw lb eps h) u

theorem checked_forward_kernel_equiv (platform : Compiler.Platform) (decoder : Decoder)
    (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule)
    (read : Nat → VeriTile.Torch.FP32.Word) (x w : VeriTile.Torch.FP32.Tensor)
    (b : Option VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x)
    (lw : VeriTile.Torch.FP32.TensorLaws w) (lb : ∀ t,b=some t → VeriTile.Torch.FP32.TensorLaws t)
    (eps : Contract.Epsilon)
    (h : Contract.InputContract x.metadata w.metadata (b.map VeriTile.Torch.FP32.Tensor.metadata) eps)
    (binding : Binding platform decoder (Contract.width x.metadata)
      (Contract.width x.metadata).nextPowerOfTwo .forward R S)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (abi : Compiler.Bound platform binding.selected .forward
      [Contract.width x.metadata,Contract.width x.metadata,Contract.width x.metadata] hostEpsilon b.isSome)
    (conversion : EpsilonConversion platform binding.selected.object P eps.value hostEpsilon)
    (u : State Term) :
    let N := Contract.width x.metadata
    let B := N.nextPowerOfTwo
    let views := forwardViews (physicalWords read) x w b lx lw lb eps h
    let bias := b.isSome
    let eps := eps.value
    let I := P binding.selected.object.primitives
    let epsilon := epsilonWord platform binding.selected.object hostEpsilon
    (KernelBound.Pointers views ∧
      ∀ schedule,eval I schedule (.literal (some .fp32) .real eps)=epsilon) ∧ ∃ v,
      FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R (withEpsilon I eps epsilon) (S "y" (u.pids 0*N+j.val)) N eps bias
          (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=
        observe (withEpsilon I eps epsilon) (S "r" (u.pids 0))
          (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o) :=
  forward_kernel_equiv platform decoder P R S (Contract.width x.metadata) (Contract.width x.metadata).nextPowerOfTwo
    binding eps.value b.isSome hostEpsilon abi conversion (forwardViews (physicalWords read) x w b lx lw lb eps h) u


theorem forward_tensor_equiv {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : Compiler.Platform) (decoder : Decoder) (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule)
    (N B : Nat) (binding : Binding platform decoder N B .forward R S) (eps : ℝ) (bias : Bool)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (_abi : Compiler.Bound platform binding.selected .forward [N,N,N] hostEpsilon bias)
    (conversion : EpsilonConversion platform binding.selected.object P eps hostEpsilon)
    (views : RegionName → Option (ReadView Term (fun a => Term.leaf "physical" a (api.read initial a))))
    (u : State Term) (_access : (KernelInputs.ForwardReads views (u.pids 0) N bias)) (outputs : KernelStorage.Placement (KernelOutputs.forwardFootprint N B (u.pids 0))) :
    let I := P binding.selected.object.primitives
    let epsilon := epsilonWord platform binding.selected.object hostEpsilon
    (KernelBound.Pointers views ∧
      ∀ schedule,eval I schedule (.literal (some .fp32) .real eps)=epsilon) ∧ ∃ v,
      (FP.Structural.exec (engine (algebra R))
        (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias) (physical views u)=some v ∧
      v.mem=KernelRows.forwardHeap (algebra R) N B eps bias (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "y" (u.pids 0*N+j.val)) (v.mem "y" (u.pids 0*N+j.val))=
        nativeForwardReference R (withEpsilon I eps epsilon) (S "y" (u.pids 0*N+j.val)) N eps bias
          (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "w" k.val) (fun k => value views u "b" k.val) j) ∧
      observe I (S "r" (u.pids 0)) (v.mem "r" (u.pids 0))=
        observe (withEpsilon I eps epsilon) (S "r" (u.pids 0))
          (KernelReference.savedInverse (algebra R) N eps (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      observe I (S "mu" (u.pids 0)) (v.mem "mu" (u.pids 0))=observe I (S "mu" (u.pids 0))
        (KernelReference.savedMean (algebra R) N (fun k : Fin B => value views u "x" (u.pids 0*N+k.val))) ∧
      (∀ q o,¬KernelOutputs.forwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o)) ∧
      KernelStorage.TensorImage api outputs (fun r i => observe I (S r i)) v initial ∧
      (∀ r i,(KernelOutputs.forwardFootprint N B (u.pids 0)) r i →
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (KernelStorage.commit api outputs (fun r i => observe I (S r i)) v initial) a⟩ : VeriTile.Torch.FP32.Word))
          (outputs.tensor r) (outputs.laws r) i).bits=observe I (S r i) (v.mem r i)) :=
  KernelStorage.nativeAttach (forward_kernel_equiv platform decoder P R S N B binding eps bias hostEpsilon _abi conversion views u)
    api outputs (fun r i => observe (P binding.selected.object.primitives) (S r i)) initial

theorem backward_tensor_equiv {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : Compiler.Platform) (decoder : Decoder) (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule)
    (N B : Nat) (binding : Binding platform decoder N B .backward R S)
    (hostEpsilon : Compiler.Shared.HostScalar)
    (_abi : Compiler.Bound platform binding.selected .backward [N,N,N,N] hostEpsilon binding.selected.object.signature.hasBias)
    (views : RegionName → Option (ReadView Term (fun a => Term.leaf "physical" a (api.read initial a))))
    (u : State Term) (_access : (KernelInputs.BackwardReads views (u.pids 0) N)) (outputs : KernelStorage.Placement (KernelOutputs.backwardFootprint N B (u.pids 0))) :
    let I := P binding.selected.object.primitives
    KernelBound.Pointers views ∧ ∃ v,
      (FP.Structural.exec (engine (algebra R))
        (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B) (physical views u)=some v ∧
      v.mem=KernelRows.backwardHeap (algebra R) N B (logical views u) ∧
      (∀ j : Fin B,j.val<N → observe I (S "dx" (u.pids 0*N+j.val)) (v.mem "dx" (u.pids 0*N+j.val))=
        nativeGradientReference R I (S "dx" (u.pids 0*N+j.val)) N (fun k => value views u "x" (u.pids 0*N+k.val))
          (fun k => value views u "dy" (u.pids 0*N+k.val)) (fun k => value views u "w" k.val)
          (value views u "mu" (u.pids 0)) (value views u "r" (u.pids 0)) j) ∧
      (∀ q o,¬KernelOutputs.backwardFootprint N B (u.pids 0) q o → v.mem q o=(physical views u).mem q o)) ∧
      KernelStorage.TensorImage api outputs (fun r i => observe I (S r i)) v initial ∧
      (∀ r i,(KernelOutputs.backwardFootprint N B (u.pids 0)) r i →
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (KernelStorage.commit api outputs (fun r i => observe I (S r i)) v initial) a⟩ : VeriTile.Torch.FP32.Word))
          (outputs.tensor r) (outputs.laws r) i).bits=observe I (S r i) (v.mem r i)) :=
  KernelStorage.nativeAttach (backward_kernel_equiv platform decoder P R S N B binding hostEpsilon _abi views u)
    api outputs (fun r i => observe (P binding.selected.object.primitives) (S r i)) initial

theorem affine_tensor_equiv {Store Byte : Type}
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API Word Store Byte) (initial : Store)
    (platform : Compiler.Platform) (decoder : Decoder) (P : PrimitiveModel) (R : Reductions) (S : OutputSchedule)
    (rows N B wc bc : Nat) (binding : Binding platform decoder N B .affineBackward R S) (bias : Bool)
    (_abi : Compiler.Bound platform binding.selected .affineBackward [rows,N,wc,bc] (.boolean Bool.false) bias)
    (views : RegionName → Option (ReadView Term (fun a => Term.leaf "physical" a (api.read initial a))))
    (u : State Term) (_access : (KernelInputs.AffineReads views rows N (u.pids 0))) (outputs : KernelStorage.Placement (KernelOutputs.affineFootprint wc bc (u.pids 0) bias)) :
    let I := P binding.selected.object.primitives
    KernelBound.Pointers views ∧ ∃ v,
      (FP.Structural.exec (engine (algebra R)) (KernelAffine.kernel rows N wc bc bias) (physical views u)=some v ∧
      v.mem=KernelAffine.heap (algebra R) rows N wc bc (u.pids 0) bias (logical views u) ∧
      (∀ j : Fin 128,u.pids 0*128+j.val<wc → observe I (S "dw" (u.pids 0*128+j.val)) (v.mem "dw" (u.pids 0*128+j.val))=
        observe I (S "dw" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (KernelBound.affineValues (algebra R) rows N views u (u.pids 0*128+j.val)).1))) ∧
      (bias=Bool.true → ∀ j : Fin 128,u.pids 0*128+j.val<bc → observe I (S "db" (u.pids 0*128+j.val)) (v.mem "db" (u.pids 0*128+j.val))=
        observe I (S "db" (u.pids 0*128+j.val)) (Cell.mk .real (KernelReference.cast (algebra R)
          (KernelBound.affineValues (algebra R) rows N views u (u.pids 0*128+j.val)).2))) ∧
      (∀ q o,¬KernelOutputs.affineFootprint wc bc (u.pids 0) bias q o → v.mem q o=(physical views u).mem q o)) ∧
      KernelStorage.TensorImage api outputs (fun r i => observe I (S r i)) v initial ∧
      (∀ r i,(KernelOutputs.affineFootprint wc bc (u.pids 0) bias) r i →
        (VeriTile.Torch.FP32.logicalValue
          (fun a => (⟨api.read (KernelStorage.commit api outputs (fun r i => observe I (S r i)) v initial) a⟩ : VeriTile.Torch.FP32.Word))
          (outputs.tensor r) (outputs.laws r) i).bits=observe I (S r i) (v.mem r i)) :=
  KernelStorage.nativeAttach (affine_kernel_equiv platform decoder P R S rows N B wc bc binding bias _abi views u)
    api outputs (fun r i => observe (P binding.selected.object.primitives) (S r i)) initial

#axiomsClean checked_forward_graph_equiv
#axiomsClean checked_forward_kernel_equiv
#axiomsClean forward_tensor_equiv
#axiomsClean backward_tensor_equiv
#axiomsClean affine_tensor_equiv
end Equivalence
end
end VeriTile.Bench.Repairs.FastLayernorm.TensorBridge
