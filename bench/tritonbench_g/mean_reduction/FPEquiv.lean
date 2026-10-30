import VeriTile.Torch.FP32
import bench.tritonbench_g.mean_reduction.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.MeanReduction.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
local instance pairWordInhabited : Inhabited VeriTile.Torch.FP32.Word := ⟨FP.Value.zero .fp32⟩

/-- The complete FP32 original grid and existing integer-index candidate are
safe, succeed, agree on every active output and preserve both exterior frames.
Only integer PID multiplication is commuted. All FP32 words, primitive models,
reduction order and complete grid permutations remain admitted. -/
specification mean_grid_fp32 (family : Nat → Algebra (FP.Value .fp32))
    (M N : Nat) (tiles : List Nat) (schedule : tiles.Perm (List.range (gridSize M))) :
    MeanIO.original family M N tiles ≡[[]] MeanIO.indexCommuted family M N tiles :=
  ProgramIO.Implementation.of_equivalent (MeanIO.candidate family M N tiles)
    (MeanIO.native_equivalent family M N tiles schedule)

#print_fp_assumptions mean_grid_fp32
#axiomsClean mean_grid_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.FPEquiv
