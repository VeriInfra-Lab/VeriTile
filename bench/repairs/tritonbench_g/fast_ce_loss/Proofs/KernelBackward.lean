import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelChunked
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelBackward

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def gradientOffset (p : Parameters) (initial : State α) (lane : TileIndex [p.block]) : Nat :=
  initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.1.val)

def activeLane (p : Parameters) (initial : State α) (lane : TileIndex [p.block]) : Prop :=
  initial.pids 1*p.block+lane.1.val<p.vocab

instance (p : Parameters) (initial : State α) : DecidablePred (activeLane p initial) :=
  fun _ => inferInstanceAs (Decidable (_<_))

def gradientMemory (M : Algebra α) (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun s lane =>
    if activeLane p initial lane then
      s.write p.logits (gradientOffset p initial lane) (.mk .real (gradientValue M p initial lane.1))
    else s) initial

/-- The complete in-place backward body succeeds, including ignored rows with
arbitrary floating-point input words. All loads precede the masked stores. -/
theorem backward_executes (M : Algebra α) (p : Parameters) (initial : State α) :
    ∃ final, exec (engine M) (backward p) initial=some final ∧
      final.mem=(gradientMemory M p initial).mem := by
  classical
  cases hc : p.hasSoftcap <;> cases hs : p.hasScaling
  all_goals
    by_cases ignored : label p initial = -100
    all_goals
      have readIgnored := ignored
      simp only [label,Region.cast] at readIgnored
      refine ⟨?_,?_,?_⟩
      rotate_left
      · simp [exec,backward,TypedSource.backward,hc,hs,ComputeKernel.surfaceBody,
          run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
          engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
          numeric,natLt,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,Region.cast,
          State.setReg,FP.Structural.store,Function.comp_def,readIgnored]
        rfl
      · simp only [gradientMemory,gradientOffset,activeLane,gradientValue,
          upstreamValue,savedLSE,backwardDerivative,backwardCorrected,
          backwardProbability,backwardScale,cap,capPartial,loaded,literal,
          KernelModel.cast,binary,unary,hc,hs,label,Region.cast,readIgnored,
          if_true,if_false,Bool.false_eq_true,decimal_one,decimal_zero]
        apply VeriTile.Bench.Repairs.CrossEntropy1.KernelNative.masked_memory_congr
        rfl

theorem gradientOffset_injective (p : Parameters) (initial : State α) :
    Function.Injective (gradientOffset p initial) := by
  intro a b same
  apply Prod.ext
  · apply Fin.ext
    simp only [gradientOffset] at same
    omega
  · exact Subsingleton.elim _ _

/-- Every active lane contains its independent ordered gradient expression;
all cells outside the active stores retain their entry values. -/
theorem backward_outputs (M : Algebra α) (p : Parameters) (initial : State α) :
    ∃ final, exec (engine M) (backward p) initial=some final ∧
      (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.vocab →
        (final.mem p.logits (gradientOffset p initial (lane,PUnit.unit))).read .real=
          gradientValue M p initial lane) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠gradientOffset p initial (lane,PUnit.unit)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,memory⟩ := backward_executes M p initial
  refine ⟨final,execution,?_,?_⟩
  · intro lane active
    rw [memory]
    have cell := State.masked_scatter_readback p.logits (gradientOffset p initial)
      (fun i => Cell.mk .real (gradientValue M p initial i.1)) (activeLane p initial)
      initial (lane,PUnit.unit) active (gradientOffset_injective p initial)
    change ((gradientMemory M p initial).mem p.logits _).read .real=_
    rw [show (gradientMemory M p initial).mem p.logits (gradientOffset p initial (lane,PUnit.unit))=
      Cell.mk .real (gradientValue M p initial lane) from cell]
    simp
  · intro region offset miss
    rw [memory]
    apply State.masked_scatter_frame
    by_cases same : region=p.logits
    · right
      intro lane _ active
      exact (miss lane.1 active).resolve_left (not_not_intro same)
    · exact Or.inl same

#axiomsClean backward_executes
#axiomsClean backward_outputs
#specNonCircular gradientValue avoiding [forward,chunked,backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
