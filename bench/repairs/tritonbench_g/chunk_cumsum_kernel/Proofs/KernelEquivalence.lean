import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Rewrite

/-! Actual FP32 source versus the displayed sum-left carry-add variant.
Both executions succeed and preserve the full exterior frame. ADD-COMMUTE is
used only for carry values read by another output chunk; the unused final
carry and unspecified tail operands need no finite-value restriction. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.KernelEquivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α : Type} [Inhabited α]

abbrev original := repairedKernel .fp32 .fp32
abbrev sumLeft := sumLeftKernel .fp32 .fp32
abbrev rewrite_domain := @Rewrite.FiniteCarry

theorem memory_equiv (ops : FPRow.Ops α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] ops.algebra D)
    (input output : RegionName) (T : Nat) (c : Config) (u : State α)
    (finite : rewrite_domain ops.algebra D input output T c.block u) :
    ∃ a b, FP.Structural.exec (engine ops.algebra) (original input output T c.block) u=some a ∧
      FP.Structural.exec (engine ops.algebra) (sumLeft input output T c.block) u=some b ∧
      a.mem=b.mem ∧ FPRow.Frame output (u.pids 0) T u a ∧ FPRow.Frame output (u.pids 0) T u b :=
  Rewrite.successful_pair ops D model input output T c.block u finite

theorem original_source (input output : RegionName) (T BT : Nat) :
    original input output T BT=repairedKernel .fp32 .fp32 input output T BT := rfl

theorem candidate_source (input output : RegionName) (T BT : Nat) :
    sumLeft input output T BT=sumLeftKernel .fp32 .fp32 input output T BT := rfl

#print_fp_assumptions Rewrite.body_equiv
#axiomsClean memory_equiv
#axiomsClean original_source
#axiomsClean candidate_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.KernelEquivalence
