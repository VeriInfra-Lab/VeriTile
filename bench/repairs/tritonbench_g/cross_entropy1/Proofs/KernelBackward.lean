import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelNative
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def inputLSE (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  M.fp32Load ((initial.mem p.lse (p.lseBase+initial.pids 0)).read .real)

def inputUpstream (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  if target p initial=p.ignored then M.literal (some .fp32) .real 0.0 else
    M.fp32Load ((initial.mem p.upstream
      (p.upstreamBase+initial.pids 0*p.upstreamStride)).read .real)

/-- Independent source-ordered backward lane graph. The ignored selection
remains outside the multiplication, so arbitrary NaNs cannot replace zero. -/
def gradientValue (M : Algebra α) (p : Parameters) (initial : State α)
    (lane : Fin p.block) : α :=
  let lit := M.literal (some .fp32) .real
  let op := M.binary (some .fp32) .real
  let lse := inputLSE M p initial
  let probability := M.unary (some .fp32) .exp
    (op .sub (logits M p initial (lane,PUnit.unit)) lse)
  let corrected := op .add probability
    (op .mul (op .mul (op .mul (lit 2.0) (RuntimeArgs.fp initial "lse_square_scale")) lse) probability)
  let shifted := (if target p initial=p.ignored then p.classStart else target p initial)-p.classStart
  let hit := ((initial.pids 1*p.block+lane.val : Nat) : Int)=shifted
  let adjusted := if p.hasSmoothing then
      op .sub (if hit then op .sub corrected (op .sub (lit 1) (RuntimeArgs.fp initial "smoothing")) else corrected)
        (op .div (RuntimeArgs.fp initial "smoothing") (M.fromInt (some .fp32) (RuntimeArgs.classes initial)))
    else if hit then op .sub corrected (lit 1.0) else corrected
  if target p initial=p.ignored then lit 0.0 else
    op .mul (inputUpstream M p initial) adjusted

def gradientOffset (p : Parameters) (initial : State α) (lane : TileIndex [p.block]) : Nat :=
  p.gradientBase+initial.pids 0*p.gradientStride+(initial.pids 1*p.block+lane.1.val)

def activeLane (p : Parameters) (initial : State α) (lane : TileIndex [p.block]) : Prop :=
  initial.pids 1*p.block+lane.1.val<p.cols

instance (p : Parameters) (initial : State α) : DecidablePred (activeLane p initial) :=
  fun _ => inferInstanceAs (Decidable (_<_))

def gradientMemory (M : Algebra α) (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [p.block]).foldl (fun s lane =>
    if activeLane p initial lane then
      s.write p.gradient (gradientOffset p initial lane) (.mk .real (gradientValue M p initial lane.1))
    else s) initial

theorem masked_memory_congr {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s t : State α) (memory : s.mem=t.mem) :
    (lanes.foldl (fun u lane => if active lane then u.write r (off lane) (val lane) else u) s).mem=
    (lanes.foldl (fun u lane => if active lane then u.write r (off lane) (val lane) else u) t).mem := by
  induction lanes generalizing s t with
  | nil => exact memory
  | cons a rest ih =>
    simp only [List.foldl_cons]
    by_cases h : active a
    · simp only [h,if_true]
      apply ih
      simp only [State.write,memory]
    · simp only [h,if_false]
      exact ih s t memory

/-- Complete successful backward source execution and exact masked memory.
This also allows gradient and logits to share their allocation. -/
theorem backward_executes (M : Algebra α) (p : Parameters) (initial : State α)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, exec (engine M) (backwardSyntax p) initial=some final ∧
      final.mem=(gradientMemory M p initial).mem := by
  cases hs : p.hasSmoothing
  all_goals
    by_cases ignored : target p initial=p.ignored
    all_goals
      simp only [target,Region.cast] at ignored
      refine ⟨?_,?_,?_⟩
      rotate_left
      · simp [exec,backwardSyntax,hs,ComputeKernel.surfaceBody,run,step,
          evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
          Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
          numeric,numericEq,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
          State.setReg,FP.Structural.store,Region.cast,ignored,abi.smoothing,abi.squareScale,abi.totalClasses]
        rfl
      · simp only [gradientMemory,gradientValue,inputLSE,inputUpstream,logits,
          hs,target,Region.cast,ignored,if_true,if_false,Bool.false_eq_true,gradientOffset,activeLane,
          RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.squareScale,abi.totalClasses,Option.getD_some]
        apply masked_memory_congr
        rfl

theorem gradientOffset_injective (p : Parameters) (initial : State α) :
    Function.Injective (gradientOffset p initial) := by
  intro a b same
  apply Prod.ext
  · apply Fin.ext
    simp only [gradientOffset] at same
    omega
  · exact Subsingleton.elim _ _

/-- Every active lane reads back its computed gradient, and every cell outside
the active stores is framed, including padding and unrelated typed regions. -/
theorem backward_outputs (M : Algebra α) (p : Parameters) (initial : State α)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, exec (engine M) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
        (final.mem p.gradient (gradientOffset p initial (lane,PUnit.unit))).read .real=
          gradientValue M p initial lane) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠gradientOffset p initial (lane,PUnit.unit)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,memory⟩ := backward_executes M p initial values abi
  refine ⟨final,execution,?_,?_⟩
  · intro lane active
    rw [memory]
    have cell := State.masked_scatter_readback p.gradient (gradientOffset p initial)
      (fun i => Cell.mk .real (gradientValue M p initial i.1)) (activeLane p initial)
      initial (lane,PUnit.unit) active (gradientOffset_injective p initial)
    change ((gradientMemory M p initial).mem p.gradient _).read .real=_
    rw [show (gradientMemory M p initial).mem p.gradient (gradientOffset p initial (lane,PUnit.unit))=
      Cell.mk .real (gradientValue M p initial lane) from cell]
    simp
  · intro region offset miss
    rw [memory]
    apply State.masked_scatter_frame
    by_cases same : region=p.gradient
    · right
      intro lane _ active
      exact (miss lane.1 active).resolve_left (not_not_intro same)
    · exact Or.inl same

#axiomsClean backward_outputs

#axiomsClean backward_executes
#specNonCircular gradientValue avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelNative
