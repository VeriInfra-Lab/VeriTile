import bench.tritonbench_g.fused_layernorm_triton.Proofs.FPNormalize
import bench.tritonbench_g.fused_layernorm_triton.Proofs.FPWelfordSoftware
namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPCorrect
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (Post)
open LayerNormWelfold.FPStages (column rowAddress normalizeValues roundedNormalize roundingInitial witnessOne)
open FPStages
noncomputable section

specification normalize_source_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (normalizeValues M s mean rstd X W Bias rnumel RBLOCK) := by
  exact FPStages.normalize_source_fp32 M mean rstd X W Bias Y rnumel RBLOCK s

specification normalize_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (roundedNormalize cfg s mean rstd X W Bias rnumel RBLOCK) := by
  exact FPStages.normalize_software_fp32 rest cfg mean rstd X W Bias Y rnumel RBLOCK s

specification normalize_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (normalizeSourceChunk "rstd" "x" "weight" "bias" "mean" "out" 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  exact FPStages.normalize_nonempty_rounding rest

/-- Both branches of the pinned helper, including its general update face. -/
specification welford_helper_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (firstIteration : Bool) (value : Op .real []) (v : α)
    (previous : WelfordTriple (Op .real [])) (prev : WelfordTriple α)
    (hv : evalOp M (some .fp32) value s = some (fun _ => v))
    (hp : EvaluatesTriple M s previous prev) :
    EvaluatesTriple M s (reduceExpressions firstIteration value previous) (reduceStep M firstIteration v prev) := by
  exact reduce_expressions_eval M s firstIteration value v previous prev hv hp

specification welford_reduction_fp32 {α : Type} [Inhabited α] (M : Algebra α) (equal32 : α→α→Bool)
    (X Mean Rstd : RegionName) (xnumel rnumel RBLOCK : Nat) (tree : ReductionTree RBLOCK)
    (hpositive : 0<rnumel) (hblock : rnumel≤RBLOCK) (htree : tree.Complete) (s : State α) :
    ∃ final, FP.Structural.exec (withEquality M equal32) (welfordBlock X Mean Rstd xnumel rnumel RBLOCK tree) s = some final ∧
      final.mem = (statisticsMemory M s Mean Rstd rnumel
        (treeValue M equal32 (firstLaneValues M s X rnumel RBLOCK) tree)).mem := by
  exact welford_block_fp32 M equal32 X Mean Rstd xnumel rnumel RBLOCK tree hpositive hblock htree s

specification welford_software_with_rsqrt [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (X Mean Rstd : RegionName)
    (xnumel rnumel N : Nat) (tree : ReductionTree N)
    (hpositive : 0<rnumel) (hblock : rnumel≤N) (htree : tree.Complete)
    (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (withEquality (LayerNormWelfold.FPStages.fp32WithRsqrt rest cfg rsqrt32) (FP.eq cfg))
      (welfordBlock X Mean Rstd xnumel rnumel N tree) s = some final ∧
      final.mem = (roundedStatistics cfg rsqrt32 s Mean Rstd rnumel
        (roundedTree cfg (roundedFirstLanes s X rnumel N) tree)).mem := by
  exact FPStages.welford_software_with_rsqrt rest cfg rsqrt32 X Mean Rstd xnumel rnumel N tree hpositive hblock htree s

specification welford_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    ∃ final, FP.Structural.exec (withEquality
      (LayerNormWelfold.FPStages.fp32WithRsqrt rest .ieeeValue rsqrt32) (FP.eq .ieeeValue))
      (welfordBlock "x" "mean" "rstd" 1 2 2 pairTree) welfordInitial = some final ∧
      final.mem "mean" 0 = .mk .real welfordOne ∧
      final.mem "mean" 1 = welfordInitial.mem "mean" 1 ∧
      (1 + 1/16777216 : Rat) ≠ 1 := by
  exact FPStages.welford_nonempty_rounding rest rsqrt32

specification welford_bit_examples :
    roundedCombine .ieeeValue
      ⟨FP.Value.ofBits .fp32 0x3f800000,FP.Value.zero .fp32,welfordOne⟩
      ⟨FP.Value.ofBits .fp32 0x3f800001,FP.Value.zero .fp32,welfordOne⟩ =
      ⟨welfordOne,FP.Value.ofBits .fp32 0x28000000,FP.Value.ofBits .fp32 0x40000000⟩ ∧
    roundedCombine .ieeeValue
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩ =
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,FP.Value.ofBits .fp32 0x40000000⟩ := by
  exact ⟨rounded_combine_midpoint,rounded_combine_infinity_guard⟩

#axiomsClean welford_helper_fp32
#axiomsClean welford_reduction_fp32
#axiomsClean welford_software_with_rsqrt
#axiomsClean welford_nonempty_rounding
#axiomsClean welford_bit_examples
#axiomsClean normalize_source_fp32
#axiomsClean normalize_software_fp32
#axiomsClean normalize_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPCorrect
