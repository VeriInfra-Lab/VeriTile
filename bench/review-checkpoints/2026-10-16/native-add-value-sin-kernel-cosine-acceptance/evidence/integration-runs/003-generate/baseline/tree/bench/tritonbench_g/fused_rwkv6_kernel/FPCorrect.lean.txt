import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.FPExpPrimitive
import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.FPShape
/-! Public stage correctness with finite precision. Exp remains an explicit
native-primitive interface in state contracts. The independent output stage
also has a closed software bit-pattern profile and nonempty rounding witness.
The conditional final stage computes an actual update before its write; it
does not assume a precomputed buffer equals the final answer. -/
namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPCorrect
open VeriTile.Triton FP.Structural FPStages
open FusedRecurrentDelta.FPStages (sequentialFP32 stateScatter outputAddress valueIndex Post)
noncomputable section
specification state_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (stateScatter s BHOut K V BK BV (stateValues M s BHPrev k v w t s_k_h s_v_h K V BK BV)).mem := by
  exact FPStages.state_fp32_stage M BHPrev k v w BHOut t s_k_h s_v_h K V BK BV s

#axiomsClean state_fp32_stage

specification output_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale) s = some final ∧
      Post s final o BV (outputAddress s t s_v_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale) := by
  exact FPStages.output_fp32_stage M BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale s

#axiomsClean output_fp32_stage

specification conditional_final_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev k v w ht : RegionName) (STORE_FINAL_STATE : Bool)
    (t s_k_h s_v_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (finalStep BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (if STORE_FINAL_STATE then
        stateScatter s ht K V BK BV (stateValues M s BHPrev k v w t s_k_h s_v_h K V BK BV) else s).mem := by
  exact FPStages.conditional_final_fp32_stage M BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV s

#axiomsClean conditional_final_fp32_stage

specification state_fp32_with_exp_primitive [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (fp32WithExp rest cfg exp32)
      (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (stateScatter s BHOut K V BK BV
        (roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV)).mem := by
  exact FPStages.state_fp32_with_exp_primitive rest cfg exp32 BHPrev k v w BHOut t s_k_h s_v_h K V BK BV s

#axiomsClean state_fp32_with_exp_primitive

specification conditional_final_fp32_with_exp_primitive [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (BHPrev k v w ht : RegionName) (STORE_FINAL_STATE : Bool)
    (t s_k_h s_v_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (fp32WithExp rest cfg exp32)
      (finalStep BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (if STORE_FINAL_STATE then stateScatter s ht K V BK BV
        (roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV) else s).mem := by
  exact FPStages.conditional_final_fp32_with_exp_primitive rest cfg exp32 BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV s

#axiomsClean conditional_final_fp32_with_exp_primitive

specification output_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV 1) s = some final ∧
      Post s final o BV (outputAddress s t s_v_h B H V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedOutput cfg s BHPrev q k v u t s_k_h s_v_h H K V BK BV) := by
  exact FPStages.output_software_fp32 rest cfg BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV s

#axiomsClean output_software_fp32

specification output_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest .ieeeValue)
      (outputStep "state" "query" "key" "value" "bias" "out" 0 1 1 1 1 1 1 1 1 2 1)
      roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  exact FPStages.output_nonempty_rounding rest

#axiomsClean output_nonempty_rounding

end
end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPCorrect
