import bench.tritonbench_g.fused_layernorm_triton.Proofs.FPWelfordBlock
import bench.tritonbench_g.layer_norm_welfold.Proofs.FPRsqrtPrimitive
namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
open VeriTile.Triton FP.Structural
open LayerNormWelfold.FPStages (fp32WithRsqrt epsilon32)
open FusedRecurrentDelta.FPStages (sequentialFP32)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

deriving instance DecidableEq for WelfordTriple

def welfordOne : FP.Value .fp32 := .ofBits .fp32 0x3f800000

/-- Independently specified bit-value operations. Every arithmetic node
rounds separately; both equality guards use ScalarOps floating equality. -/
def roundedCombine (cfg : FP.Config) (l r : WelfordTriple (FP.Value .fp32)) :
    WelfordTriple (FP.Value .fp32) :=
  let z := FP.Value.zero .fp32
  let delta := if FP.eq cfg l.mean r.mean then z else FP.sub cfg r.mean l.mean
  let w := FP.add cfg l.weight r.weight
  let ratio := if FP.eq cfg w z then z else FP.div cfg r.weight w
  ⟨FP.add cfg l.mean (FP.mul cfg delta ratio),
    FP.add cfg (FP.add cfg l.m2 r.m2)
      (FP.mul cfg (FP.mul cfg (FP.mul cfg delta delta) l.weight) ratio), w⟩

def roundedTree (cfg : FP.Config) {N : Nat} (values : Fin N → WelfordTriple (FP.Value .fp32)) :
    ReductionTree N → WelfordTriple (FP.Value .fp32)
  | .leaf i => values i
  | .node l r => roundedCombine cfg (roundedTree cfg values l) (roundedTree cfg values r)

def roundedFirstLanes [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32))
    (X : RegionName) (rnumel N : Nat) : Fin N → WelfordTriple (FP.Value .fp32) := fun i =>
  if i.val < rnumel then
    ⟨(s.mem X (i.val+rnumel*s.pids 0)).read .real, FP.Value.zero .fp32,welfordOne⟩
  else ⟨FP.Value.zero .fp32,FP.Value.zero .fp32,FP.Value.zero .fp32⟩

def roundedStatistics (cfg : FP.Config) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (s : State (FP.Value .fp32)) (Mean Rstd : RegionName) (rnumel : Nat)
    (v : WelfordTriple (FP.Value .fp32)) : State (FP.Value .fp32) :=
  (s.write Mean (s.pids 0) (.mk .real v.mean)).write Rstd (s.pids 0)
    (.mk .real (rsqrt32 (FP.add cfg (FP.div cfg v.m2 (FP.round .fp32 cfg (rnumel : Rat))) epsilon32)))

theorem software_combine (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (l r : WelfordTriple (FP.Value .fp32)) :
    combineTriple (fp32WithRsqrt rest cfg rsqrt32) (FP.eq cfg) l r = roundedCombine cfg l r := by
  simp [combineTriple,roundedCombine,fp32WithRsqrt,sequentialFP32,
    FP.Software.scalarFP32,FP.Software.fp32Context]
  norm_num

theorem software_tree (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) {N : Nat}
    (values : Fin N → WelfordTriple (FP.Value .fp32)) (tree : ReductionTree N) :
    treeValue (fp32WithRsqrt rest cfg rsqrt32) (FP.eq cfg) values tree = roundedTree cfg values tree := by
  induction tree with
  | leaf i => rfl
  | node l r hl hr => simp only [treeValue,roundedTree,hl,hr,software_combine]

/-- Scalar arithmetic, equality, conversion and constants are concrete fp32.
The sole numerical callback used in the result is the explicit rsqrt32. -/
specification welford_software_with_rsqrt [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (X Mean Rstd : RegionName)
    (xnumel rnumel N : Nat) (tree : ReductionTree N)
    (hpositive : 0<rnumel) (hblock : rnumel≤N) (htree : tree.Complete)
    (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (withEquality (fp32WithRsqrt rest cfg rsqrt32) (FP.eq cfg))
      (welfordBlock X Mean Rstd xnumel rnumel N tree) s = some final ∧
      final.mem = (roundedStatistics cfg rsqrt32 s Mean Rstd rnumel
        (roundedTree cfg (roundedFirstLanes s X rnumel N) tree)).mem := by
  obtain ⟨final,he,hm⟩ := welford_block_fp32 (fp32WithRsqrt rest cfg rsqrt32) (FP.eq cfg)
    X Mean Rstd xnumel rnumel N tree hpositive hblock htree s
  refine ⟨final,he,?_⟩
  have lanes : firstLaneValues (fp32WithRsqrt rest cfg rsqrt32) s X rnumel N =
      roundedFirstLanes s X rnumel N := by
    funext i
    simp [firstLaneValues,roundedFirstLanes,welfordOne,fp32WithRsqrt,sequentialFP32,
      FP.Software.scalarFP32,FP.Software.fp32Context]
    norm_num
  rw [lanes,software_tree] at hm
  simpa [statisticsMemory,roundedStatistics,rstdValue,fp32WithRsqrt,sequentialFP32,
    FP.Software.scalarFP32,FP.Software.fp32Context] using hm

/-- A nontrivial two-leaf combine: the exact mean is 1+2^-24, while binary32
nearest-even gives 1. This is neither identity rounding nor a real formula. -/
theorem rounded_combine_midpoint :
    roundedCombine .ieeeValue
      ⟨FP.Value.ofBits .fp32 0x3f800000,FP.Value.zero .fp32,welfordOne⟩
      ⟨FP.Value.ofBits .fp32 0x3f800001,FP.Value.zero .fp32,welfordOne⟩ =
      ⟨welfordOne,FP.Value.ofBits .fp32 0x28000000,FP.Value.ofBits .fp32 0x40000000⟩ := by
  decide +kernel

/-- Infinity-equality guard is material: equal infinities yield zero delta,
so the tuple's m2 is zero instead of the NaN from infinity-infinity. -/
theorem rounded_combine_infinity_guard :
    roundedCombine .ieeeValue
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩ =
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,FP.Value.ofBits .fp32 0x40000000⟩ := by
  decide +kernel

theorem first_iteration_infinity (rest : Algebra (FP.Value .fp32)) :
    reduceStep (FP.Software.scalarFP32 rest .ieeeValue) Bool.true (FP.Value.infinity .fp32)
      ⟨FP.Value.zero .fp32,FP.Value.zero .fp32,FP.Value.zero .fp32⟩ =
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩ := by
  simp [reduceStep,FP.Software.scalarFP32,FP.Software.fp32Context,welfordOne]

def pairTree : ReductionTree 2 := .node (.leaf ⟨0,by decide⟩) (.leaf ⟨1,by decide⟩)

theorem pair_tree_complete : pairTree.Complete := by
  simp [ReductionTree.Complete,ReductionTree.leaves,pairTree,List.finRange_succ]

def welfordInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun _ off => .mk .real (if off=1 then FP.Value.ofBits .fp32 0x3f800001 else welfordOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem initial_pair_value [Inhabited (FP.Value .fp32)] :
    roundedTree .ieeeValue (roundedFirstLanes welfordInitial "x" 2 2) pairTree =
      ⟨welfordOne,FP.Value.ofBits .fp32 0x28000000,FP.Value.ofBits .fp32 0x40000000⟩ := by
  simpa [pairTree,roundedTree,roundedFirstLanes,welfordInitial,Cell.read,welfordOne] using rounded_combine_midpoint

/-- Nonempty execution and memory/frame witness. Two lanes are an interpreter
instance, not a claim that this is a native autotune RBLOCK configuration. -/
specification welford_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    ∃ final, FP.Structural.exec (withEquality (fp32WithRsqrt rest .ieeeValue rsqrt32) (FP.eq .ieeeValue))
      (welfordBlock "x" "mean" "rstd" 1 2 2 pairTree) welfordInitial = some final ∧
      final.mem "mean" 0 = .mk .real welfordOne ∧
      final.mem "mean" 1 = welfordInitial.mem "mean" 1 ∧
      (1 + 1/16777216 : Rat) ≠ 1 := by
  obtain ⟨final,he,hm⟩ := welford_software_with_rsqrt rest .ieeeValue rsqrt32
    "x" "mean" "rstd" 1 2 2 pairTree (by decide) (by decide) pair_tree_complete welfordInitial
  refine ⟨final,he,?_,?_,by norm_num⟩
  · rw [hm,initial_pair_value]
    simp [roundedStatistics,State.write,welfordInitial]
  · rw [hm]
    simp [roundedStatistics,State.write,welfordInitial]

#axiomsClean software_combine
#axiomsClean software_tree
#axiomsClean welford_software_with_rsqrt
#axiomsClean rounded_combine_midpoint
#axiomsClean rounded_combine_infinity_guard
#axiomsClean first_iteration_infinity
#axiomsClean pair_tree_complete
#axiomsClean initial_pair_value
#axiomsClean welford_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
