import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_ForwardFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_ForwardMath

/-! Independent draft: successful complete fp32 forward execution for an
arbitrary structural algebra and initial state. The actual source produces
all registers from that state. This is a success statement, not a numerical
formula, memory-typing guarantee, native execution claim or added premise of
the existing empty-assumption equivalence theorem.

The current source lowers maximum and sum to the real carrier. Their algebra
callbacks are total; the maximum still checks its positive axis extent.
There is no assumed whole-run equation or prepared intermediate state.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedTotal
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- An actual original-source final state for all eight flags and the three
admitted mask modes, with arbitrary regions, strides, K and initial state.
Only positive tile depth is required; unknown strings are not in MaskMode. -/
def fp32_source_run {α : Type} [Inhabited α] (algebra : Algebra α)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (initial : State α) :
    {final : State α // run (engine algebra)
      (Typed.fp32Original Y X M ym yn xm xn sm K D
        log mode.python causal fp16).surfaceBody initial = some final} := by
  cases mode <;> cases log <;> cases causal <;> cases fp16
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [Typed.fp32Original, MaskMode.python,
      ComputeKernel.surfaceBody, run, step, stepAlg,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
      ComputeDType.eraseDType, numeric, bop, numericLt, numericLe, natLt, natLe,
      ofFloat, toFloat, TileShape.axisDim, hD, Region.cast, State.setReg,
      FP.Structural.store]
    try rfl

/-- The candidate shares the original's produced state. Equivalence is
used only after original execution success has been established above. -/
theorem fp32_nonempty_pair {α : Type} [Inhabited α] (algebra : Algebra α)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (initial : State α) :
    ∃ final : State α, ∀ commuted : Bool,
      run (engine algebra)
        (Typed.checkedKernel .fp32 commuted Y X M ym yn xm xn sm K D
          log mode.python causal fp16).surfaceBody initial = some final := by
  let witness := fp32_source_run algebra Y X M ym yn xm xn sm K D hD
    log mode causal fp16 initial
  refine ⟨witness.val, ?_⟩
  intro commuted
  cases commuted
  · exact witness.property
  · exact (Floating.forward_run_eq .fp32 Y X M ym yn xm xn sm K D
      log mode.python causal fp16 algebra initial).symm.trans witness.property

end
#axiomsClean fp32_source_run
#axiomsClean fp32_nonempty_pair
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedTotal
