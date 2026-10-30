import bench.repairs.tritonbench_g.f8_conversion_utils.Kernels
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.F8ConversionUtils.KernelNative
open VeriTile.Torch Checked
noncomputable section

/-- The full shared typed kernel implements the independent scalar formula;
both decoder stores and the complete typed-cell frame are retained. -/
theorem complete_kernel_bit_formula (policy : Numerics.Policy) (p : Source.Profile)
    (Y X : VeriTile.Triton.RegionName) (N B : Nat)
    (initial : VeriTile.Triton.FP.Structural.State Source.BitAlgebra.Scalar)
    (words : Nat → Bytes.Word p.inputWidth)
    (inputs : KernelContract.InputWords p X N B initial words) :
    ∃ final, VeriTile.Triton.FP.Structural.exec
        ((KernelContract.model policy p).withDefaultPrecision .fp32)
        (Source.kernel p Y X N B) initial = some final ∧
      (∀ j : Fin B, KernelContract.active initial N B j →
        Source.BitAlgebra.unpack p.outputWidth
          (Source.outputValue p (final.mem Y (initial.pids 0*B+j.val))) =
          conversionBits policy p (words (initial.pids 0*B+j.val))) ∧
      (∀ r i, (r≠Y ∨ ∀ j : Fin B, KernelContract.active initial N B j →
        i≠initial.pids 0*B+j.val) → final.mem r i=initial.mem r i) :=
  KernelContract.complete policy p Y X N B initial words inputs

/-- Successful source executions refine the independent finite native graph
at every actual launch; the source Tensor count has no added finite guard. -/
theorem complete_native_kernel_refinement (policy : Numerics.Policy) (p : Source.Profile)
    (source : Native.Allocation n p.inputWidth) (target : Native.Allocation n p.outputWidth) :
    FPCalls p (conversionBits policy p) source target :=
  KernelContract.native_calls policy p source target

theorem complete_native_byte_frame (policy : Numerics.Policy) (p : Source.Profile)
    (input output count pid : Nat) (before : Bytes.Heap) (q : Nat)
    (outside : ∀ lane, lane<1024 → NativeIndex.mask count pid lane=Bool.true →
      q<NativeIndex.pointer output p.outputWidth pid lane ∨
        NativeIndex.pointer output p.outputWidth pid lane+p.outputWidth≤q) :
    Lowering.program (Bytes.primitives p.inputWidth p.outputWidth) p.twice
      (conversionBits policy p) input output count pid before q=before q :=
  KernelContract.native_program_frame policy p input output count pid before q outside

/-- Primitive cast/NaN/rounding policy belongs to the defined machine semantics.
There are no optimization laws requiring numerical experiments in this task. -/
def deferredNumericalObligations : List String := []
#axiomsClean complete_kernel_bit_formula
#axiomsClean complete_native_kernel_refinement
#axiomsClean complete_native_byte_frame

end
end VeriTile.Bench.Repairs.F8ConversionUtils.KernelNative
