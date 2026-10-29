import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialFP32 word stateAddress keyAddress valueAddress keyIndex valueIndex stateScatter)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000

/-- ScalarOps supplies the exact finite-precision add/mul/casts. Exp remains
an explicit primitive on binary32 bit patterns. Binding this callback to a
native instruction/library is a separate obligation, not an algebraic law. -/
def fp32WithExp (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (exp32 : FP.Value .fp32 → FP.Value .fp32) : Algebra (FP.Value .fp32) :=
  { sequentialFP32 rest cfg with
    unary := fun p op x =>
      match p, op with
      | some .fp32, .exp => exp32 x
      | _, _ => rest.unary p op x }

def roundedStateWithExp [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (exp32 : FP.Value .fp32 → FP.Value .fp32) (s : State (FP.Value .fp32))
    (BHPrev k v w : RegionName) (t s_k_h s_v_h K V BK BV : Nat) : TileIndex [BV,BK] → FP.Value .fp32 := fun i =>
  let previous := word s BHPrev (stateAddress s K V BK BV i)
    (valueIndex s BV i.1 < V ∧ keyIndex s BK i.2.1 < K)
  let key := word s k (keyAddress s t s_k_h K BK i.2) (keyIndex s BK i.2.1 < K)
  let val := word s v (valueAddress s t s_v_h V BV (i.1,PUnit.unit)) (valueIndex s BV i.1 < V)
  let wval := word s w (keyAddress s t s_k_h K BK i.2) (keyIndex s BK i.2.1 < K)
  FP.add cfg (FP.mul cfg previous (exp32 wval)) (FP.mul cfg key val)

theorem stateValues_fp32WithExp [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (s : State (FP.Value .fp32)) (BHPrev k v w : RegionName) (t s_k_h s_v_h K V BK BV : Nat) :
    stateValues (fp32WithExp rest cfg exp32) s BHPrev k v w t s_k_h s_v_h K V BK BV =
      roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV := by
  funext i
  simp [stateValues, previousValue, keyValue, valueValue, roundedStateWithExp,
    fp32WithExp, sequentialFP32, FusedRecurrentDelta.FPStages.maskedInput, word,
    FP.Software.scalarFP32, FP.Software.fp32Context]

specification state_fp32_with_exp_primitive [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (fp32WithExp rest cfg exp32)
      (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (stateScatter s BHOut K V BK BV
        (roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV)).mem := by
  obtain ⟨final, he, hm⟩ := state_fp32_stage (fp32WithExp rest cfg exp32) BHPrev k v w BHOut t s_k_h s_v_h K V BK BV s
  exact ⟨final, he, by simpa only [stateValues_fp32WithExp] using hm⟩

specification conditional_final_fp32_with_exp_primitive [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (BHPrev k v w ht : RegionName) (STORE_FINAL_STATE : Bool)
    (t s_k_h s_v_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (fp32WithExp rest cfg exp32)
      (finalStep BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (if STORE_FINAL_STATE then stateScatter s ht K V BK BV
        (roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV) else s).mem := by
  obtain ⟨final, he, hm⟩ := conditional_final_fp32_stage (fp32WithExp rest cfg exp32)
    BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV s
  exact ⟨final, he, by simpa only [stateValues_fp32WithExp] using hm⟩

#axiomsClean state_fp32_with_exp_primitive
#axiomsClean conditional_final_fp32_with_exp_primitive
end
end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
