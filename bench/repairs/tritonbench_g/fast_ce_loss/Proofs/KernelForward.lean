import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelModel
import Mathlib.Tactic.NormNum.OfScientific

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem negative_block_offset (a b c : Nat) :
    (-(a : Int) + -((b : Int) * c)).toNat = 0 := by
  have positive : (0 : Int) ≤ (b : Int) * c := Int.mul_nonneg (by omega) (by omega)
  omega

@[simp] theorem natural_row_address (row stride : Nat) :
    ((row : Int)*(stride : Int)).toNat=row*stride := by norm_cast

@[simp] theorem decimal_zero : (0.0 : ℝ)=0 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num

@[simp] theorem decimal_one : (1.0 : ℝ)=1 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  have h : mkRat (↑(10 : Nat)) (10^1) = (1 : ℚ) := by decide
  simpa only [Rat.cast_one] using congrArg (fun q : ℚ => (q : ℝ)) h

def forwardMemory (M : Algebra α) (p : Parameters) (initial : State α) : State α :=
  (initial.write p.lse (initial.pids 0) (.mk .real (lseValue M p initial 0))).write
    p.loss (initial.pids 0) (.mk .real (forwardLoss M p initial))

/-- Complete source execution derives the final memory from the entry state.
The caller supplies only the active reduction width and ordinary label range. -/
theorem forward_executes (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (validLabel : label p initial = -100 ∨ 0≤label p initial) :
    ∃ final, exec (engine M) (forward p) initial=some final ∧
      final.mem=(forwardMemory M p initial).mem := by
  classical
  cases hc : p.hasSoftcap <;> cases hs : p.hasScaling
  all_goals
    by_cases ignored : label p initial = -100
    all_goals
      have readIgnored := ignored
      simp only [label,Region.cast] at readIgnored
      refine ⟨?_,?_,?_⟩
      rotate_left
      · simp [exec,forward,TypedSource.forward,hc,hs,ComputeKernel.surfaceBody,
          run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
          engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
          numeric,natLt,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,Region.cast,
          State.setReg,FP.Structural.store,TileShape.axisDim,positive,Function.comp_def,
          readIgnored]
        rfl
      · funext region offset
        simp only [forwardMemory,forwardLoss,ignored,if_true,if_false]
        dsimp only [State.write,lseValue,lseTile,maxValue,maxTile,forwardLogits,loaded,
          forwardScale,cap,capPartial,literal,KernelModel.cast,binary,unary,gathered,label]
        simp only [forwardLogits,loaded,forwardScale,cap,capPartial,literal,
          KernelModel.cast,binary,unary,hc,hs,Bool.false_eq_true,↓reduceIte,Nat.zero_mul,Nat.zero_add,
          bop,Broadcast.leftIndex,Broadcast.rightIndex]
        rfl

end
end VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
