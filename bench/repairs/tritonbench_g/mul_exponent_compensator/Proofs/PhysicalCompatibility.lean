import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.PhysicalIO

/-! Original finite-address statements as optional ordinary consequences. The
source IO and finite signed-index lowering are connected by PhysicalIO. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.PhysicalCompatibility
open VeriTile.Triton
open VeriTile.Bench.Repairs.TritonMul2 (Memory Layout Allocated)
noncomputable section

theorem kernel_real (l : Layout)
    (blocks : l.block∈supportedBlocks) (gridBound : grid l.count l.block ≤ 2^31-1)
    (allocation : Allocated l) (order : List Nat)
    (schedule : order.Perm (List.range (grid l.count l.block))) (before : Memory ℝ) :
    ∃ final,Native.sourceLaunch ⟨KernelArithmetic.mathematical⟩ l order before=some final ∧
      (∀ i,i<l.count → final (l.output+i)=KernelArithmetic.mathematical (before (l.input+i))) ∧
      (∀ cell,(¬∃ i,i<l.count ∧ cell=l.output+i) → final cell=before cell) :=
  PhysicalIO.complete _ _ KernelArithmetic.scalar_real l blocks gridBound allocation order schedule before

theorem kernel_fp_profile (profile : FP.Config) (l : Layout)
    (blocks : l.block∈supportedBlocks) (gridBound : grid l.count l.block ≤ 2^31-1)
    (allocation : Allocated l) (order : List Nat)
    (schedule : order.Perm (List.range (grid l.count l.block))) (before : Memory KernelArithmetic.FPWord) :
    ∃ final,Native.sourceLaunch ⟨KernelArithmetic.native profile⟩ l order before=some final ∧
      (∀ i,i<l.count → final (l.output+i)=KernelArithmetic.native profile (before (l.input+i))) ∧
      (∀ cell,(¬∃ i,i<l.count ∧ cell=l.output+i) → final cell=before cell) :=
  PhysicalIO.complete _ _ (KernelArithmetic.scalar_native profile) l blocks gridBound allocation order schedule before

theorem kernel_fp (l : Layout)
    (blocks : l.block∈supportedBlocks) (gridBound : grid l.count l.block ≤ 2^31-1)
    (allocation : Allocated l) (order : List Nat)
    (schedule : order.Perm (List.range (grid l.count l.block))) (before : Memory KernelArithmetic.FPWord) :
    ∃ final,Native.sourceLaunch ⟨KernelArithmetic.native KernelArithmetic.selectedProfile⟩ l order before=some final ∧
      (∀ i,i<l.count → final (l.output+i)=KernelArithmetic.native KernelArithmetic.selectedProfile (before (l.input+i))) ∧
      (∀ cell,(¬∃ i,i<l.count ∧ cell=l.output+i) → final cell=before cell) :=
  kernel_fp_profile KernelArithmetic.selectedProfile l blocks gridBound allocation order schedule before

#axiomsClean kernel_real
#axiomsClean kernel_fp_profile
#axiomsClean kernel_fp
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.PhysicalCompatibility
