import bench.tritonbench_g.decay_cumsum.IO
import bench.tritonbench_g.decay_cumsum.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.DecayCumsum.Correct
open VeriTile Triton FP.Structural Geometry
open scoped VeriTile.Spec
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The exact decimal-weighted inclusive prefix at each logical time/feature. -/
def forwardPrefix (h : Call) (p : Phase) (xs : LogicalData.Matrix h p) : LogicalData.Matrix h p :=
  fun t j => ∑ i ∈ Finset.range (t.val+1), LogicalData.sample xs i j * 1.44269504

/-- Preparation preserves the source's q/k gates and last-time exponent. -/
def prepareResult (h : Call) (p : Phase) (scale : ℝ) (xs : LogicalData.PrepareInputs h p) :
    LogicalData.PrepareOutputs h p :=
  { qg := fun t j => xs.q t j * ((2 : ℝ)^(xs.g t j) * scale)
    kg := fun t j => xs.k t j * (2 : ℝ)^(LogicalData.sample xs.g (p.span-1) j - xs.g t j) }

/-- Independent saved dq/dk inputs, in-place dq/dk results and reverse prefix. -/
def backwardResult (h : Call) (p : Phase) (xs : LogicalData.BackwardInputs h p) :
    LogicalData.BackwardOutputs h p :=
  let dq := fun t j => LogicalData.sample xs.dqi t j + LogicalData.sample xs.dqo t j *
    (2 : ℝ)^(LogicalData.sample xs.g t j)
  let dk := fun t j => LogicalData.sample xs.dki t j + LogicalData.sample xs.dko t j *
    (2 : ℝ)^(LogicalData.sample xs.g (p.span-1) j - LogicalData.sample xs.g t j)
  { dq := fun t j => dq t.val j
    dk := fun t j => dk t.val j
    dg := fun t j => ∑ i ∈ Finset.range (p.span-t.val),
      (dq (p.span-1-i) j * LogicalData.sample xs.q (p.span-1-i) j -
        dk (p.span-1-i) j * LogicalData.sample xs.k (p.span-1-i) j) }

def stageTarget (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    LogicalData.Inputs kind h p → LogicalData.Outputs kind h p :=
  match kind with
  | .forward => forwardPrefix h p
  | .prepare => prepareResult h p scale
  | .backward => backwardResult h p

/-- Each complete selected main/tail program implements its independent
forward prefix, gated Q/K or backward reverse-prefix formula. Explicit IO
retains every emitted time/lane, in-place dqo/dko, source access safety,
successful execution and the full raw-cell exterior frame. -/
specification decay_cumsum_kernel_real (rest : Algebra ℝ) (kind : Kind)
    (h : Call) (p : Phase) (hp : p∈phases h) (prog : Program h p) (scale : ℝ) :
    Spec.Real ((LogicalData.io rest kind h p prog scale) ⊨
      (stageTarget kind h p scale)) :=
  LogicalProof.correct rest kind h p hp prog scale

#specNonCircular stageTarget avoiding [Kernels.source,FP.Structural.exec,FPLogical.evaluate]
#axiomsClean decay_cumsum_kernel_real

#guard_msgs (drop info) in
#auditModuleAxioms
#stmtSurfaceSubset decay_cumsum_kernel_real ⊆
  [VeriTile.Bench.Repairs.DecayCumsum.Call,
   VeriTile.Bench.Repairs.DecayCumsum.Correct.stageTarget,
   VeriTile.Bench.Repairs.DecayCumsum.Geometry.Program,
   VeriTile.Bench.Repairs.DecayCumsum.Kind,
   VeriTile.Bench.Repairs.DecayCumsum.LogicalData.Inputs,
   VeriTile.Bench.Repairs.DecayCumsum.LogicalData.Outputs,
   VeriTile.Bench.Repairs.DecayCumsum.LogicalData.io,
   VeriTile.Bench.Repairs.DecayCumsum.Phase,
   VeriTile.Bench.Repairs.DecayCumsum.phases,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.Algebra,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.Implements]

end
end VeriTile.Bench.Repairs.DecayCumsum.Correct
