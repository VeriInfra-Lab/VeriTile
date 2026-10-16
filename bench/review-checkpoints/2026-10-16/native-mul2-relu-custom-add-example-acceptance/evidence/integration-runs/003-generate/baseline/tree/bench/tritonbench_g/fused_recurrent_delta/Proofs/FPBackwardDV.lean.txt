import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardValues
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem dvHeadwise_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV scale)).mem := by
  simp [dvHeadwise, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dvHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dvHeadwise_memory M DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dv BV _ _ _ (outputAddress_injective s t s_vo_h B H V BV) hm⟩

#axiomsClean dvHeadwise_fp32_stage

theorem dvScalar_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV scale)).mem := by
  simp [dvScalar, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dvScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dvScalar_memory M DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dv BV _ _ _ (outputAddress_injective s t s_vo_h B H V BV) hm⟩

#axiomsClean dvScalar_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
