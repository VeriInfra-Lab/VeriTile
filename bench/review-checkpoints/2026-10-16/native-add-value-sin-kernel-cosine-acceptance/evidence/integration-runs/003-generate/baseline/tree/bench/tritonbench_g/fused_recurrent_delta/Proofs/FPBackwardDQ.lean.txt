import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardValues
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem dqHeadwise_memory {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV scale)).mem := by
  simp [dqHeadwise, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dqHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dqHeadwise_memory M HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dq BK _ _ _ (keyOutputAddress_injective s t s_qk_h B H K BK) hm⟩

#axiomsClean dqHeadwise_fp32_stage

theorem dqScalar_memory {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV scale)).mem := by
  simp [dqScalar, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dqScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dqScalar_memory M HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dq BK _ _ _ (keyOutputAddress_injective s t s_qk_h B H K BK) hm⟩

#axiomsClean dqScalar_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
