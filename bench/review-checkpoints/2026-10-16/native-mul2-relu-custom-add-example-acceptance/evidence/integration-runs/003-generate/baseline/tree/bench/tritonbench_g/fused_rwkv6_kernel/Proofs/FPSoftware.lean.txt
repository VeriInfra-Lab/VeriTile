import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialFP32 sequentialSum word stateAddress keyAddress valueAddress outputAddress keyIndex valueIndex Post)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000

/-- The output stage does not evaluate exp: it consumes the current state
before the update. Its concrete profile therefore requires only ScalarOps
and the explicitly ordered fp32 reduction. -/
def roundedOutput [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (s : State (FP.Value .fp32))
    (BHPrev q k v u : RegionName) (t s_k_h s_v_h H K V BK BV : Nat) : TileIndex [BV] → FP.Value .fp32 :=
  sequentialSum cfg (shape := [BV,BK]) ⟨1, by simp⟩ Bool.false (fun i =>
    let previous := word s BHPrev (stateAddress s K V BK BV i)
      (valueIndex s BV i.1 < V ∧ keyIndex s BK i.2.1 < K)
    let key := word s k (keyAddress s t s_k_h K BK i.2) (keyIndex s BK i.2.1 < K)
    let val := word s v (valueAddress s t s_v_h V BV (i.1,PUnit.unit)) (valueIndex s BV i.1 < V)
    let bias := word s u (uAddress s H K BK i.2.1) (keyIndex s BK i.2.1 < K)
    let query := FP.mul cfg (word s q (keyAddress s t s_k_h K BK i.2) (keyIndex s BK i.2.1 < K))
      (FP.Value.ofBits .fp32 0x3f800000)
    FP.mul cfg (FP.add cfg previous (FP.mul cfg (FP.mul cfg key val) bias)) query)

specification output_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV 1) s = some final ∧
      Post s final o BV (outputAddress s t s_v_h B H V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedOutput cfg s BHPrev q k v u t s_k_h s_v_h H K V BK BV) := by
  obtain ⟨final, he, hp⟩ := output_fp32_stage (sequentialFP32 rest cfg)
    BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV 1 s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [outputValues, previousValue, keyValue, valueValue, roundedOutput,
    sequentialFP32, FusedRecurrentDelta.FPStages.maskedInput, word,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessQuarterUlp : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_rounding :
    FP.add .ieeeValue (FP.Value.zero .fp32)
      (FP.mul .ieeeValue
        (FP.add .ieeeValue witnessOne
          (FP.mul .ieeeValue (FP.mul .ieeeValue witnessQuarterUlp witnessOne) witnessOne))
        (FP.mul .ieeeValue witnessOne witnessOne)) = witnessOne ∧
      witnessOne.toRat? = some 1 ∧ witnessQuarterUlp.toRat? = some (1 / 33554432) ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

def roundingInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun R _ => .mk .real
    (if R = ("key" : RegionName) ∨ R = ("out" : RegionName) then witnessQuarterUlp else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

specification output_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest .ieeeValue)
      (outputStep "state" "query" "key" "value" "bias" "out" 0 1 1 1 1 1 1 1 1 2 1)
      roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := output_software_fp32 rest .ieeeValue
    "state" "query" "key" "value" "bias" "out" 0 1 1 1 1 1 1 1 1 2 roundingInitial
  refine ⟨final, he, ?_, ?_, witness_rounding.2.2.2⟩
  · have hv := hp.1 (⟨0, by decide⟩, PUnit.unit) (by simp [valueIndex, roundingInitial])
    exact hv.trans (congrArg (Cell.mk .real) witness_rounding.1)
  · apply hp.2 "out" 1
    right
    intro i hi
    simp [valueIndex, roundingInitial] at hi
    simp [outputAddress, roundingInitial]
    omega

#axiomsClean output_software_fp32
#axiomsClean witness_rounding
#axiomsClean output_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
