import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedTotal
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedModel

/-! Complete typed execution success for all four checked source profiles.
All source controls, casts, operations
and the original elif without a final else remain in the quoted kernels.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedTotal
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 16000

/-- Produce the complete original typed run from arbitrary initial state;
no intermediate registers or callback results are provided by the caller. -/
def checked_source_run {α : Type} [Inhabited α] (algebra : Algebra α)
    (profile : Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (initial : State α) :
    {final : State α // run (engine algebra)
      (Typed.checkedKernel profile Bool.false Y X M ym yn xm xn sm K D
        log mode.python causal fp16).surfaceBody initial = some final} := by
  cases profile
  · exact fp32_source_run algebra Y X M ym yn xm xn sm K D hD
      log mode causal fp16 initial
  all_goals
    cases mode <;> cases log <;> cases causal <;> cases fp16
    all_goals
      refine ⟨?_, ?_⟩
      swap
      simp (maxSteps := 500000) [Typed.checkedKernel,
        Typed.narrowOriginal, Typed.mixedOriginal, Typed.wideOriginal, MaskMode.python,
        ComputeKernel.surfaceBody, run, step, stepAlg,
        FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
        Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision,
        ComputeDType.eraseDType, numeric, bop, numericLt, numericLe, natLt, natLe,
        ofFloat, toFloat, TileShape.axisDim, hD, Region.cast, State.setReg,
        FP.Structural.store]
      try rfl

/-- All four checked profiles, three admitted mask modes, every source
flag, both variants, positive D, and arbitrary algebra/state. This concerns
structural success only; mismatched initial memory channels retain the
interpreter's existing default-read behavior, not a numeric refinement. -/
theorem checked_nonempty_pair {α : Type} [Inhabited α] (algebra : Algebra α)
    (profile : Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (initial : State α) :
    ∃ final : State α, ∀ commuted : Bool,
      run (engine algebra)
        (Typed.checkedKernel profile commuted Y X M ym yn xm xn sm K D
          log mode.python causal fp16).surfaceBody initial = some final := by
  let witness := checked_source_run algebra profile Y X M ym yn xm xn sm K D hD
    log mode causal fp16 initial
  refine ⟨witness.val, ?_⟩
  intro commuted
  cases commuted
  · exact witness.property
  · exact (Floating.forward_run_eq profile Y X M ym yn xm xn sm K D
      log mode.python causal fp16 algebra initial).symm.trans witness.property

/-- The existing extended-real witness model is one instance of the same
success theorem. No replacement algebra or numerical premise is introduced. -/
theorem checked_model_nonempty (profile : Typed.CheckedProfile)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool)
    (initial : State (WithBot ℝ)) :
    ∃ final : State (WithBot ℝ), ∀ commuted : Bool,
      run (engine TypedExecution.model)
        (Typed.checkedKernel profile commuted Y X M ym yn xm xn sm K D
          log mode.python causal fp16).surfaceBody initial = some final :=
  checked_nonempty_pair TypedExecution.model profile Y X M ym yn xm xn sm K D hD
    log mode causal fp16 initial

end
#axiomsClean checked_source_run
#axiomsClean checked_nonempty_pair
#axiomsClean checked_model_nonempty
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedTotal
