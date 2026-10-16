import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardValues
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem dkHeadwise_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV scale)).mem := by
  simp [dkHeadwise, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dkHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dkHeadwise_memory M DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dk BK _ _ _ (keyOutputAddress_injective s t s_qk_h B H K BK) hm⟩

#axiomsClean dkHeadwise_fp32_stage

theorem dkScalar_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV scale)).mem := by
  simp [dkScalar, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dkScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dkScalar_memory M DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dk BK _ _ _ (keyOutputAddress_injective s t s_qk_h B H K BK) hm⟩

#axiomsClean dkScalar_fp32_stage

theorem dkCorrection_memory {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      final.mem = (scatter s dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (correctionValues M s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV)).mem := by
  simp [dkCorrection, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dkCorrection_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (correctionValues M s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) := by
  obtain ⟨final, he, hm⟩ := dkCorrection_memory M HPrev dv dk t s_qk_h s_vo_h B H K V BK BV s
  exact ⟨final, he, post_of_memory s final dk BK _ _ _ (keyOutputAddress_injective s t s_qk_h B H K BK) hm⟩

#axiomsClean dkCorrection_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
