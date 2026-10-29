import bench.tritonbench_g.logsumexp_fwd.Proofs.FPShape
import bench.tritonbench_g.softmax_reducev.Proofs.FPSoftware

namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd.WordGraph
open VeriTile.Triton FP.Structural FP.Equational
open SoftmaxReducev.FPValue (treeReduce maximumWord)
open FPStages (rowBlock inputAddress outputAddress rowValue)
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32

/-- The two transcendental maps and the host scalar conversion remain explicit.
The original general Real scale domain is not replaced by scale=1. Reduction
plans are valid finite trees, not an asserted native reduction schedule. -/
def backend (rest : Algebra Word) (cfg : FP.Config) (scalar : ℝ → Word)
    (exp32 log32 : Word → Word) (sumPlans maxPlans : ∀ n, ReductionPlan n) : Algebra Word :=
  { FP.Software.scalarFP32 rest cfg with
    literal := fun p d r => if FP.Software.fp32Context p d then scalar r else rest.literal p d r
    unary := fun p op v => match p, op with
      | some .fp32, .exp => exp32 v
      | some .fp32, .log => log32 v
      | _, _ => rest.unary p op v
    reduceMax := fun p _ ax keep xs => if p = some .fp32 then
      treeReduce maxPlans (maximumWord cfg) (FP.Value.infinity .fp32 Bool.true) ax keep xs
      else rest.reduceMax p ax keep xs
    reduceSum := fun p _ ax keep xs => if p = some .fp32 then
      treeReduce sumPlans (FP.add cfg) (FP.Value.zero .fp32) ax keep xs
      else rest.reduceSum p ax keep xs }

/-- Reads only the independent initial memory, retaining all padded -Inf lanes.
`.real` is the original explicit-fp32 computation carrier, not native fp64. -/
def loaded (s : State Word) (x : RegionName) (D B : Nat) : Values Word .real [B] :=
  fun i => if s.pids 1 * B + i.1.val < D then
    (s.mem x (inputAddress s D B i.1)).read .real
    else FP.Value.infinity .fp32 Bool.true

def scaled (cfg : FP.Config) (scalar : ℝ → Word) (scale : ℝ)
    (s : State Word) (x : RegionName) (D B : Nat) (HAS_SCALE : Bool) : Values Word .real [B] :=
  if HAS_SCALE then fun i => FP.mul cfg (loaded s x D B i) (scalar scale)
  else loaded s x D B

def exponentArguments (cfg : FP.Config) (scalar : ℝ → Word) (maxPlans : ∀ n, ReductionPlan n)
    (scale : ℝ) (s : State Word) (x : RegionName) (D B : Nat) (HAS_SCALE : Bool) : Values Word .real [B] :=
  let xs := scaled cfg scalar scale s x D B HAS_SCALE
  let maximum := treeReduce maxPlans (maximumWord cfg) (FP.Value.infinity .fp32 Bool.true)
    ⟨0, by simp⟩ Bool.false xs ()
  fun i => FP.sub cfg (xs i) maximum

def output (cfg : FP.Config) (scalar : ℝ → Word) (exp32 log32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (scale : ℝ)
    (s : State Word) (x : RegionName) (D B : Nat) (HAS_SCALE : Bool) : Word :=
  let xs := scaled cfg scalar scale s x D B HAS_SCALE
  let maximum := treeReduce maxPlans (maximumWord cfg) (FP.Value.infinity .fp32 Bool.true)
    ⟨0, by simp⟩ Bool.false xs ()
  let exps := fun i => exp32 (FP.sub cfg (xs i) maximum)
  let total := treeReduce sumPlans (FP.add cfg) (FP.Value.zero .fp32) ⟨0, by simp⟩ Bool.false exps ()
  FP.add cfg (log32 total) maximum

/-- Exact whole-graph interpretation; no equality of an execution result is a premise. -/
theorem reference_graph (rest : Algebra Word) (cfg : FP.Config) (scalar : ℝ → Word)
    (exp32 log32 : Word → Word) (sumPlans maxPlans : ∀ n, ReductionPlan n)
    (scale : ℝ) (s : State Word) (x : RegionName) (D B : Nat) (HAS_SCALE : Bool) :
    rowValue (backend rest cfg scalar exp32 log32 sumPlans maxPlans) s x D B HAS_SCALE scale =
      output cfg scalar exp32 log32 sumPlans maxPlans scale s x D B HAS_SCALE := by
  cases HAS_SCALE <;> rfl

/-- Complete original positive-B scope, arbitrary scale, pids and initial state.
The result contains no `rest` callback; exp/log/scalar conversion remain named.
This theorem is a refined backend contract, not a closed transcendental model. -/
theorem complete_row (rest : Algebra Word) (cfg : FP.Config) (scalar : ℝ → Word)
    (exp32 log32 : Word → Word) (sumPlans maxPlans : ∀ n, ReductionPlan n)
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) (s : State Word) (hB : 0 < B) :
    ∃ final, FP.Structural.exec (backend rest cfg scalar exp32 log32 sumPlans maxPlans)
      (rowBlock x z D B HAS_SCALE scale) s = some final ∧
      final.mem = (s.write z (outputAddress s D B)
        (.mk .real (output cfg scalar exp32 log32 sumPlans maxPlans scale s x D B HAS_SCALE))).mem := by
  obtain ⟨final, he, hm⟩ := FPStages.row_block_fp32
    (backend rest cfg scalar exp32 log32 sumPlans maxPlans) x z D B HAS_SCALE scale s hB
  refine ⟨final, he, ?_⟩
  simpa only [reference_graph] using hm

theorem complete_row_readback_frame (rest : Algebra Word) (cfg : FP.Config) (scalar : ℝ → Word)
    (exp32 log32 : Word → Word) (sumPlans maxPlans : ∀ n, ReductionPlan n)
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) (s : State Word) (hB : 0 < B) :
    ∃ final, FP.Structural.exec (backend rest cfg scalar exp32 log32 sumPlans maxPlans)
      (rowBlock x z D B HAS_SCALE scale) s = some final ∧
      final.mem z (outputAddress s D B) = .mk .real
        (output cfg scalar exp32 log32 sumPlans maxPlans scale s x D B HAS_SCALE) ∧
      (∀ R o, R ≠ z ∨ o ≠ outputAddress s D B → final.mem R o = s.mem R o) := by
  obtain ⟨final, he, hm⟩ := complete_row rest cfg scalar exp32 log32 sumPlans maxPlans
    x z D B HAS_SCALE scale s hB
  refine ⟨final,he,?_,?_⟩
  · rw [hm]; simp
  · intro R o h
    rw [hm]
    exact State.write_other s z R (outputAddress s D B) o _ h

#axiomsClean reference_graph
#axiomsClean complete_row
#axiomsClean complete_row_readback_frame
end
end VeriTile.Bench.TritonBenchG.LogsumexpFwd.WordGraph
