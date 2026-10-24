import bench.tritonbench_g.logsumexp_fwd.Proofs.WordGraph
import bench.tritonbench_g.token_softmax_bloom.Proofs.ExpLoweringGap

namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd.WordGraph
open VeriTile.Triton FP.Structural FP.Equational
open SoftmaxReducev.FPValue (treeReduce maximumWord)
open TokenSoftmaxBloom.ExpLoweringGap
  (inputA inputB log2e sharedArgument roundedExpA roundedExpB
   input_decoding_and_subtraction no_ex2_function_matches_both)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Only the exact scale=1 fixture uses this scalar conversion profile.
This is not a proposed conversion for arbitrary Real host scalars. -/
def fixtureScalar (r : ℝ) : Word :=
  if r = 1 then FP.Value.ofBits .fp32 0x3f800000 else FP.Value.qNaN .fp32

def probeInitial (x : Word) : State Word where
  mem := fun r o => if r = "x" then .mk .real
      (if o = 35 then x else FP.Value.zero .fp32)
    else .mk .real (FP.Value.ofBits .fp32 0xdeadbeef)
  regs := fun _ _ _ => none
  pids := fun i => if i = 0 then 2 else 0
  numPids := fun _ => 3
  undef := fun d _ _ => defaultValue d

/-- D=17, B=32 is a legal original host specialization. Row 2 has seventeen
active finite lanes and fifteen padded -Inf lanes. No half-subnormal input is used. -/
def probeRow (x : Word) (i : TileIndex [32]) : Word :=
  if i.1.val < 17 then
    if i.1.val = 1 then x else FP.Value.zero .fp32
  else FP.Value.infinity .fp32 Bool.true

theorem loaded_probe (x : Word) :
    loaded (probeInitial x) "x" 17 32 = probeRow x := by
  funext i
  simp [loaded,probeInitial,FPStages.inputAddress,probeRow,Cell.read]
  split_ifs <;> (try simp_all)
  all_goals omega

theorem finite_and_padding_mul_one :
    FP.mul .ieeeValue inputA (fixtureScalar 1) = inputA ∧
    FP.mul .ieeeValue inputB (fixtureScalar 1) = inputB ∧
    FP.mul .ieeeValue (FP.Value.zero .fp32) (fixtureScalar 1) = FP.Value.zero .fp32 ∧
    FP.mul .ieeeValue (FP.Value.infinity .fp32 Bool.true) (fixtureScalar 1) =
      FP.Value.infinity .fp32 Bool.true := by
  simp only [fixtureScalar,if_pos rfl]
  decide +kernel

theorem scaled_probe (flag : Bool) :
    scaled .ieeeValue fixtureScalar 1 (probeInitial inputA) "x" 17 32 flag = probeRow inputA ∧
    scaled .ieeeValue fixtureScalar 1 (probeInitial inputB) "x" 17 32 flag = probeRow inputB := by
  cases flag
  · simp [scaled,loaded_probe]
  · simp only [scaled,Bool.true_eq,if_true,loaded_probe]
    constructor <;> funext i
    all_goals simp only [probeRow]
    all_goals split_ifs
    all_goals first
      | exact finite_and_padding_mul_one.1
      | exact finite_and_padding_mul_one.2.1
      | exact finite_and_padding_mul_one.2.2.1
      | exact finite_and_padding_mul_one.2.2.2

def probeMaximum (x : Word) : Word :=
  treeReduce ReductionPlan.withSeed (maximumWord .ieeeValue)
    (FP.Value.infinity .fp32 Bool.true) ⟨0,by simp⟩ Bool.false (probeRow x) ()

theorem probe_maximums :
    probeMaximum inputA = FP.Value.zero .fp32 ∧
    probeMaximum inputB = FP.Value.zero .fp32 := by
  decide +kernel

/-- The original load/optional scale/maximum/subtract prefix reaches the two
previously certified exp collision inputs, for both original flag branches. -/
theorem source_reaches_both_inputs (flag : Bool) :
    exponentArguments .ieeeValue fixtureScalar ReductionPlan.withSeed 1
      (probeInitial inputA) "x" 17 32 flag (⟨1,by decide⟩,()) = inputA ∧
    exponentArguments .ieeeValue fixtureScalar ReductionPlan.withSeed 1
      (probeInitial inputB) "x" 17 32 flag (⟨1,by decide⟩,()) = inputB := by
  simp only [exponentArguments,(scaled_probe flag).1,(scaled_probe flag).2]
  change FP.sub .ieeeValue (probeRow inputA (⟨1,by decide⟩,())) (probeMaximum inputA) = inputA ∧
    FP.sub .ieeeValue (probeRow inputB (⟨1,by decide⟩,())) (probeMaximum inputB) = inputB
  rw [probe_maximums.1,probe_maximums.2]
  simpa only [probeRow,if_pos (by decide : 1 < 17),if_pos rfl] using
    input_decoding_and_subtraction.2.2

/-- The observed fusion-on scaled prefix is FMA(scale,loaded,-maximum).
Only these exact two fixtures are related here; no universal contraction law. -/
theorem fixture_fused_prefix_reaches_inputs :
    FP.fma .ieeeValue (fixtureScalar 1) inputA (probeMaximum inputA).neg = inputA ∧
    FP.fma .ieeeValue (fixtureScalar 1) inputB (probeMaximum inputB).neg = inputB := by
  rw [probe_maximums.1,probe_maximums.2]
  simp only [fixtureScalar,if_pos rfl]
  decide +kernel

/-- Reuses the pinned Real.exp nearest-cell and multiply-collision proof.
This is an internal-stage refutation only, not a final stored-LSE counterexample. -/
theorem reachable_exp_binding_impossible (flag : Bool) (sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue (exponentArguments .ieeeValue fixtureScalar ReductionPlan.withSeed 1
        (probeInitial inputA) "x" 17 32 flag (⟨1,by decide⟩,())) log2e) = roundedExpA ∧
      sfu (FP.mul .ieeeValue (exponentArguments .ieeeValue fixtureScalar ReductionPlan.withSeed 1
        (probeInitial inputB) "x" 17 32 flag (⟨1,by decide⟩,())) log2e) = roundedExpB) := by
  rw [(source_reaches_both_inputs flag).1,(source_reaches_both_inputs flag).2]
  exact no_ex2_function_matches_both sfu

theorem fused_fixture_exp_binding_impossible (sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue (FP.fma .ieeeValue (fixtureScalar 1) inputA
        (probeMaximum inputA).neg) log2e) = roundedExpA ∧
      sfu (FP.mul .ieeeValue (FP.fma .ieeeValue (fixtureScalar 1) inputB
        (probeMaximum inputB).neg) log2e) = roundedExpB) := by
  rw [fixture_fused_prefix_reaches_inputs.1,fixture_fused_prefix_reaches_inputs.2]
  exact no_ex2_function_matches_both sfu

#axiomsClean loaded_probe
#axiomsClean finite_and_padding_mul_one
#axiomsClean scaled_probe
#axiomsClean probe_maximums
#axiomsClean source_reaches_both_inputs
#axiomsClean fixture_fused_prefix_reaches_inputs
#axiomsClean reachable_exp_binding_impossible
#axiomsClean fused_fixture_exp_binding_impossible
end
end VeriTile.Bench.TritonBenchG.LogsumexpFwd.WordGraph
