import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.LogicalResult

/-! Finite whole-tensor kernel specifications. The shared IO judgment includes
actual source access safety, complete-grid success, every tensor output and exact
exterior cells. Forward retains finite values and -∞, with one finite value per column;
backward retains the saved-output VJP. Operational statements live in Auxiliary. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Correct
open VeriTile Triton FP.Structural Interface
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

/-- Logical M × N × K forward normalization and the saved-output VJP use
exactly N entries of their logical column, independently of source launch tiles. -/
specification kernel_forward (X Y : RegionName) (separate : X≠Y)
    (d : Dimensions) (c : LaunchConfig) (hN : 0<d.N) (hK : 0<d.K)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    Spec.Real ((logicalForwardIO X Y d c) ⊨{LogicalForwardDomain X Y d c}
      fun xs row n col =>
        (((xs row n col).elim 0 Real.exp /
          ∑ j : Fin d.N, (xs row j col).elim 0 Real.exp : ℝ) : WithBot ℝ)) :=
  (logical_forward_iff X Y d c hK).mpr (forward_real X Y separate d c hN hK hB hC hone)

specification kernel_backward (P G Y : RegionName) (hPY : P≠Y) (hGY : G≠Y)
    (d : Dimensions) (c : LaunchConfig) (hN : 0<d.N) (hK : 0<d.K)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    Spec.Real ((logicalBackwardIO P G Y d c) ⊨ fun xs row n col =>
      xs.saved row n col * (xs.gradient row n col - ∑ j : Fin d.N,
        xs.saved row j col * xs.gradient row j col)) :=
  (logical_backward_iff P G Y d c hN hK).mpr (backward_real P G Y hPY hGY d c hN hK hB hC hone)

#axiomsClean kernel_forward
#axiomsClean kernel_backward
#stmtSurfaceSubset kernel_forward ⊆
  [Spec.Real,RegionName,Dimensions,Dimensions.M,Dimensions.N,Dimensions.K,LaunchConfig,LaunchConfig.B,
   LaunchConfig.C,LaunchConfig.one,oneTile,logicalForwardIO,LogicalForwardDomain,LogicalProgramIO.ImplementsOn,Tensor]
#stmtSurfaceSubset kernel_backward ⊆
  [Spec.Real,RegionName,Dimensions,Dimensions.M,Dimensions.N,Dimensions.K,LaunchConfig,LaunchConfig.B,
   LaunchConfig.C,LaunchConfig.one,oneTile,logicalBackwardIO,LogicalProgramIO.Implements,BackwardInputs,BackwardInputs.saved,BackwardInputs.gradient,Tensor]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Correct
