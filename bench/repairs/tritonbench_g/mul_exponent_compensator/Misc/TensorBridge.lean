import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.PhysicalCompatibility
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.TensorIO

/-! Optional actual Tensor pointer/byte/index bindings and empty-Tensor application.
The supporting implementation remains in Proofs/KernelTensor.lean. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.TensorBridge
open VeriTile.Triton KernelTensor
open VeriTile.Torch.FP32 (Tensor TensorLaws logicalValue)
open VeriTile.Bench.Repairs.TritonMul2 (Memory)
open TritonMul2.Host (logical logical_fp32 observeByte byte_frame)
noncomputable section

alias kernel_real := PhysicalCompatibility.kernel_real
alias kernel_fp_profile := PhysicalCompatibility.kernel_fp_profile
alias kernel_fp := PhysicalCompatibility.kernel_fp
export KernelArithmetic (program real_grid native_grid scalar_real scalar_native
  constant_exact selectedProfile)

/-- Kernel-only successful complete source execution at actual Tensor addresses,
with independent x[i]*2^112 output and a complete exterior word-memory frame.
The Real interpretation is ideal arithmetic, not a native/Real accuracy claim. -/
theorem correct (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (admit : InputContract x.metadata current B) (p : Buffers x lx current B admit)
    (before : Memory ℝ) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,Native.sourceLaunch ⟨KernelArithmetic.mathematical⟩ (layout p) order before=some final ∧
      (∀ i,i<x.metadata.numel → logical final p.output i=logical before x i*(2:ℝ)^112) ∧
      Frame p before final ∧
      (∀ i,i<x.metadata.numel → logical final x i=logical before x i) :=
  TensorIO.execute _ _ KernelArithmetic.scalar_real x lx current B admit p before order schedule

/-- All-word exact kernel result for an explicit concrete FP32 scalar profile.
Rounding, overflow, subnormals, infinities and NaNs are computed by FP.mul;
there is no arbitrary multiplication port or numerical equality hypothesis. -/
theorem fp_equiv_profile (profile : FP.Config)
    (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (admit : InputContract x.metadata current B) (p : Buffers x lx current B admit)
    (before : Memory KernelArithmetic.FPWord) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,Native.sourceLaunch ⟨KernelArithmetic.native profile⟩ (layout p) order before=some final ∧
      (∀ i,i<x.metadata.numel → logicalValue (fun a => final (a/4)) p.output p.laws i=
        KernelArithmetic.native profile (logicalValue (fun a => before (a/4)) x lx i)) ∧
      Frame p before final ∧
      (∀ address,address<4*(layout p).output ∨ 4*((layout p).output+x.metadata.numel)≤address →
        observeByte final address=observeByte before address) ∧
      (∀ i,i<x.metadata.numel → logicalValue (fun a => final (a/4)) x lx i=
        logicalValue (fun a => before (a/4)) x lx i) := by
  obtain ⟨final,execution,values,frame,input⟩ := TensorIO.execute
    _ _ (KernelArithmetic.scalar_native profile) x lx current B admit p before order schedule
  refine ⟨final,execution,?_,frame,byte_frame before final _ _ frame,?_⟩
  · intro i hi
    rw [logical_fp32 final p.output p.laws p.noNeg i,logical_fp32 before x lx admit.noNeg i]
    exact values i hi
  · intro i hi
    rw [logical_fp32 final x lx admit.noNeg i,logical_fp32 before x lx admit.noNeg i]
    exact input i hi

/-- Selected RNE/no-FTZ model, supported by pinned offline CUDA lowering.
This states model execution; GPU/compiler implementation conformance remains
an explicit boundary and is not a deferred numerical experiment. -/
theorem fp_equiv (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (admit : InputContract x.metadata current B) (p : Buffers x lx current B admit)
    (before : Memory KernelArithmetic.FPWord) (order : List Nat)
    (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,Native.sourceLaunch ⟨KernelArithmetic.native KernelArithmetic.selectedProfile⟩
        (layout p) order before=some final ∧
      (∀ i,i<x.metadata.numel → logicalValue (fun a => final (a/4)) p.output p.laws i=
        KernelArithmetic.native KernelArithmetic.selectedProfile
          (logicalValue (fun a => before (a/4)) x lx i)) ∧
      Frame p before final ∧
      (∀ address,address<4*(layout p).output ∨ 4*((layout p).output+x.metadata.numel)≤address →
        observeByte final address=observeByte before address) ∧
      (∀ i,i<x.metadata.numel → logicalValue (fun a => final (a/4)) x lx i=
        logicalValue (fun a => before (a/4)) x lx i) :=
  fp_equiv_profile KernelArithmetic.selectedProfile x lx current B admit p before order schedule

#axiomsClean correct
#axiomsClean fp_equiv_profile
#axiomsClean fp_equiv

export KernelTensor (actual_pointers actual_bytes formed_indices empty)

#axiomsClean actual_pointers
#axiomsClean actual_bytes
#axiomsClean formed_indices
#axiomsClean empty
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.TensorBridge
