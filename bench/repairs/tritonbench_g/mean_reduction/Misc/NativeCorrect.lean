import bench.repairs.tritonbench_g.mean_reduction.Proofs.NativeIO
import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.MeanReduction.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section
local instance wordInhabited : Inhabited VeriTile.Torch.FP32.Word := ⟨FP.Value.zero .fp32⟩

/-- The complete FP32 grid implements the independent source-order primitive
formula and agrees with the existing integer-index candidate. Every active
source access is safe, both schedules succeed and preserve the full exterior
cell frame. The family retains each program's chosen reduction primitive;
there is no floating rewrite assumption or restriction on FP32 words. -/
theorem mean_grid_native_formula (family : Nat → Algebra (FP.Value .fp32))
    (M N : Nat) (tiles : List Nat) (schedule : tiles.Perm (List.range (gridSize M))) :
    (MeanIO.nativeIO family Bool.false M N tiles).Implements (MeanIO.nativeTarget family M N) ∧
      (MeanIO.nativeIO family Bool.false M N tiles).Equivalent (MeanIO.candidate family M N tiles) :=
  ⟨MeanIO.native_implements family Bool.false M N tiles schedule,
   MeanIO.native_equivalent family M N tiles schedule⟩

-- The retained standard program-equivalence proof audits the same exact
-- integer transformation. Primitive backends above are model parameters.
#print_fp_assumptions KernelNativeResult.mean_index_equiv
#stmtSurfaceSubset mean_grid_native_formula ⊆
  [FP.Structural.Algebra,FP.Value,FP.Format.fp32,wordInhabited,MeanIO.nativeIO,MeanIO.nativeTarget,MeanIO.candidate,
   ProgramIO.Implements,ProgramIO.Equivalent,gridSize]
#specNonCircular MeanIO.nativeTarget avoiding
  [kernel,indexCommutedKernel,FP.Structural.exec,runSchedule]
#axiomsClean mean_grid_native_formula
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.FPEquiv
