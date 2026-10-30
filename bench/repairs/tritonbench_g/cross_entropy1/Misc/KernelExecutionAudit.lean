import bench.repairs.tritonbench_g.cross_entropy1.Kernels
import VeriTile.Triton.Float.ExecutionProfile

/-! Regression witnesses for actual typed DSL execution. These intentionally
do not import or execute `Source.forward` / `Source.backward`. The symbolic
floating backend is arbitrary. This closes syntax-success questions for these
nonzero-offset tail invocations, not the general source/refinement theorem. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelExecutionAudit
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def parameters (smoothing : Bool) (split : Bool) : Parameters where
  loss := "loss"
  lse := "lse"
  logits := "logits"
  gradient := "gradient"
  upstream := "upstream"
  labels := ⟨"labels"⟩
  lossBase := 2
  lseBase := 1
  logitsBase := 2
  gradientBase := 3
  upstreamBase := 1
  labelsBase := 1
  rows := 2
  cols := 3
  totalClasses := 3
  classStart := 0
  ignored := -100
  logitsStride := 4
  gradientStride := 4
  upstreamStride := 2
  block := 2
  smoothing := 1/4
  squareScale := 1/2
  hasSmoothing := smoothing
  split := split

def initial {α : Type} [Inhabited α] (data : RegionName → Nat → α) (ignored : Bool) : State α :=
  let s : State α := {
    mem := fun r i => if r="labels" then .mk .int (if ignored then -100 else 2)
      else .mk .real (data r i)
    regs := fun _ _ _ => none
    pids := fun _ => 1
    numPids := fun _ => 2
    undef := fun d _ _ => defaultValue d }
  let s := s.setReg "smoothing" .real [] (fun _ => data "smoothing" 0)
  let s := s.setReg "lse_square_scale" .real [] (fun _ => data "lse_square_scale" 0)
  s.setReg "total_classes" .int [] (fun _ => 3)

/-- The entire repaired backward DSL succeeds for arbitrary opaque floating
primitives and arbitrary input words, for either smoothing/ignored branch.
The invocation has nonzero row, block and storage offsets and a masked tail. -/
def backwardExecution {α : Type} [Inhabited α] (M : Algebra α)
    (data : RegionName → Nat → α) (smoothing ignored : Bool) :
    {final : State α // exec (M.withDefaultPrecision .fp32)
      (backwardSyntax (parameters smoothing Bool.true)) (initial data ignored)=some final} := by
  cases smoothing <;> cases ignored
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [exec,backwardSyntax,parameters,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,bop,initial,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store]
    rfl

#axiomsClean backwardExecution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelExecutionAudit
