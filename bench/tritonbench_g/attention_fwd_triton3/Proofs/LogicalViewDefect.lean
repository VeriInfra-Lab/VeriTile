import bench.tritonbench_g.attention_fwd_triton3.Proofs.FP32Profile
import VeriTile.Triton.Float.ProfiledRewrite

/-! Concrete original-source logical-view defect. Use initialized, finalized
FP32 attention with two query heads sharing one KV head, one query/key row,
D=16, and properly padded contiguous O/M/L. A lazy-negative V view changes
logical attention while every physical source input remains identical. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.LogicalViewDefect
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 20000

structure ValueView where
  physical : Nat → ℝ
  negative : Bool

def plain : ValueView := ⟨fun _ => 1, Bool.false⟩
def negative : ValueView := ⟨fun _ => 1, Bool.true⟩

/-- The independent natural-exp attention target on a logical V view. Q and K
are zero, and the signed window (offset=0,size=64) contains the single key.
The formula does not mention source execution, buffers, blocks or outputs. -/
def target (v : ValueView) : ℝ :=
  (∑ j : Fin 1, Real.exp ((∑ _d : Fin 16, (0 : ℝ) * 0) * (1/4)) *
    (if v.negative then -v.physical j.val else v.physical j.val)) /
  ∑ _j : Fin 1, Real.exp ((∑ _d : Fin 16, (0 : ℝ) * 0) * (1/4))

theorem target_plain : target plain = 1 := by norm_num [target, plain]
theorem target_negative : target negative = -1 := by norm_num [target, negative]

/-- Actual selected FP32 source for the complete (1,2) grid. Query shape is
[1,2,1,16], KV shape [1,1,1,16], O shape [1,2,64,16], M/L [1,2,64].
INIT=END=1; both boundary checks are enabled. No carry or empty-row premise. -/
def source : ComputeKernel := FP32Source.original "Q" "K" "V" "M" "Out" "L" (1/4)
  32 16 16 1 16 16 16 1 16 16 16 1 2048 1024 16 1
  1 2 1 1 64 1 0 0 64 16 64 1 1 1 0 0 64

def rawState (v : ValueView) : State ℝ where
  mem := fun rg off => if rg = "V" then .mk .fp32 (v.physical off) else .mk .fp32 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax = 1 then 2 else 1
  undef := fun d _ _ => defaultValue d

/-- Execute both actual program indices, then observe logical row 0/channel 0
of the returned slice. Out itself is an ordinary, nonnegative FP32 tensor, so
slice/contiguous/to(fp32) preserves this coordinate. -/
def observed (M : Algebra ℝ) (v : ValueView) : Option ℝ :=
  ((FP.Structural.exec (engine M) source (rawState v)).bind fun first =>
    FP.Structural.exec (engine M) source
      { first with pids := fun ax => if ax = 1 then 1 else 0 }).map
      (fun final => (final.mem "Out" 0).read .fp32)

theorem same_complete_source_execution (M : Algebra ℝ) :
    observed M plain = observed M negative := rfl

/-- No deterministic numerical backend can implement both logical contracts.
There is no premise about an assumed result, no native GPU claim, and no
floating-point accuracy assumption. This refutes the original API target. -/
theorem original_logical_contract_impossible (M : Algebra ℝ) :
    ¬ (observed M plain = some (target plain) ∧
      observed M negative = some (target negative)) := by
  intro h
  have he : (some (target plain) : Option ℝ) = some (target negative) :=
    h.1.symm.trans ((same_complete_source_execution M).trans h.2)
  have hv := Option.some.inj he
  rw [target_plain, target_negative] at hv
  norm_num at hv

#specNonCircular target avoiding [observed, rawState, source, FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.LogicalViewDefect
