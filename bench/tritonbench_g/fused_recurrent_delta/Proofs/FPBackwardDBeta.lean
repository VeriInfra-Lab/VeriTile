import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardValues
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter scatter_memory_congr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem dbetaHeadwise_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale) s = some final ∧
      final.mem = (scatter s dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (dbetaValues M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale)).mem := by
  simp [dbetaHeadwise, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  apply scatter_memory_congr
  rfl

specification dbetaHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale) s = some final ∧
      Post s final dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (dbetaValues M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale) := by
  obtain ⟨final, he, hm⟩ := dbetaHeadwise_memory M DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale s
  exact ⟨final, he, post_of_memory s final dbeta BV _ _ _ (betaOutputAddress_injective s t s_vo_h NK B H V BV) hm⟩

#axiomsClean dbetaHeadwise_fp32_stage

theorem dbetaScalar_memory {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (scalarDbetaValue M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale))).mem := by
  simp [dbetaScalar, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, natLt, numericLt, Region.cast, dkValues, dvValues, dbetaValues, scalarDbetaValue,
    rawDbetaValues, dqValues, correctionValues, dhValues, rawDvValues, loadKey, loadValue, loadBeta,
    loadBackState, maskedInput, scatter, backStateAddress, keyAddress, valueAddress,
    keyOutputAddress, outputAddress, betaOutputAddress, scalarBetaAddress, keyIndex, valueIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  simp [State.write]
  rfl

specification dbetaScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (scalarDbetaValue M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale))).mem := by
  exact dbetaScalar_memory M DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale s

#axiomsClean dbetaScalar_fp32_stage
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
