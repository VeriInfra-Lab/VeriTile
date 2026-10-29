import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LogicalMath
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.DecayCumsum.LogicalProof
open VeriTile.Triton FP.Structural Geometry
noncomputable section
set_option maxHeartbeats 2000000

/-- Every old role/channel cell is preserved, including in-place dq/dk outputs. -/
theorem target_transport (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (fun raw => (LogicalData.outputEquiv kind h p).symm
      (LogicalMath.stageTarget kind h p scale (LogicalData.inputEquiv kind h p raw))) =
      DecayIO.realTarget kind h p scale := by
  cases kind <;> funext raw o j <;> fin_cases o <;> rcases j with ⟨t,j,u⟩ <;> cases u
  all_goals
    dsimp [LogicalData.outputEquiv, LogicalData.inputEquiv,
      LogicalData.input_forward,LogicalData.input_prepare,LogicalData.input_backward,
      LogicalData.output_forward,LogicalData.output_prepare,LogicalData.output_backward,
      LogicalMath.stageTarget,LogicalMath.forwardPrefix,LogicalMath.prepareResult,
      LogicalMath.backwardResult,LogicalData.sample,DecayIO.realTarget,
      DecayIO.channelResult,DecayIO.outputIndex,DecayIO.logicalRead,
      DecayIO.outputs, Fin.cases, KernelReal.evaluate,KernelReal.forward,
      KernelReal.prepareQ,KernelReal.prepareK,KernelReal.backwardQ,KernelReal.backwardK,
      KernelReal.backward,KernelReal.gradient,KernelReal.exp2]
    first
    | rfl
    | change (show ℝ from raw (0 : Fin 3) (t,j,())) *
          ((2 : ℝ)^(show ℝ from raw (2 : Fin 3) (t,j,())) * scale) =
        (if ht : t.val < p.span then (show ℝ from raw (0 : Fin 3) (⟨t.val,ht⟩,j,())) else 0) *
          ((2 : ℝ)^(if ht : t.val < p.span then (show ℝ from raw (2 : Fin 3) (⟨t.val,ht⟩,j,())) else 0) * scale)
      simp only [dif_pos t.isLt]
    | change (show ℝ from raw (1 : Fin 3) (t,j,())) * (2 : ℝ)^(
        (if ht : p.span-1 < p.span then (show ℝ from raw (2 : Fin 3) (⟨p.span-1,ht⟩,j,())) else 0) -
          (show ℝ from raw (2 : Fin 3) (t,j,()))) =
        (if ht : t.val < p.span then (show ℝ from raw (1 : Fin 3) (⟨t.val,ht⟩,j,())) else 0) * (2 : ℝ)^(
          (if ht : p.span-1 < p.span then (show ℝ from raw (2 : Fin 3) (⟨p.span-1,ht⟩,j,())) else 0) -
            (if ht : t.val < p.span then (show ℝ from raw (2 : Fin 3) (⟨t.val,ht⟩,j,())) else 0))
      simp only [dif_pos t.isLt]

/-- Exact equivalence of full IO judgments at each original selected program. -/
theorem logical_iff (rest : Algebra ℝ) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) :
    (LogicalData.io rest kind h p prog scale).Implements
      (LogicalMath.stageTarget kind h p scale) ↔
      (DecayIO.io (KernelReal.algebra rest) kind h p prog scale).Implements
        (DecayIO.realTarget kind h p scale) :=
  Iff.of_eq (congrArg (DecayIO.io (KernelReal.algebra rest) kind h p prog scale).Implements
    (target_transport kind h p scale))

theorem correct (rest : Algebra ℝ) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : ℝ) :
    (LogicalData.io rest kind h p prog scale).Implements
      (LogicalMath.stageTarget kind h p scale) :=
  (logical_iff rest kind h p prog scale).mpr (IOCorrect.real rest kind h p hp prog scale)

end
end VeriTile.Bench.Repairs.DecayCumsum.LogicalProof
