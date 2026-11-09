import bench.tritonbench_g.spinning_lock_reduction.Proofs.LogicalIO
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Real

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Correct
open VeriTile VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
open Classical
open scoped VeriTile.Triton.LogicalDomainIO

/-- Independent finite stage formula. Producers publish zero partials and
ready locks; each stage zeros only its active output coordinates. Other input
cells are retained, including holes and overlapping output-stride locations. -/
def stageFormula (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (input : Formula.Values ℝ h count) : Formula.Values ℝ h count := {
    partials := fun i =>
      if stage = .producers ∧ (i.val / (h.bm*h.bn)) % h.groupK ≠ 0 then 0 else input.partials i
    output := fun i =>
      if ∃ p < h.numSms,
          (match stage with | .producers => p % h.groupK ≠ 0 | .consumers => p % h.groupK = 0) ∧
          ∃ row < h.bm, ∃ col < h.bn, row < h.m ∧ p*h.bn+col < h.n ∧
            row*h.strideM+(p*h.bn+col)*h.strideN = i.val
      then 0 else input.output i
    locks := fun i => if stage = .producers ∧ i.val % h.groupK ≠ 0 then 1 else input.locks i }

/-- Both original source stages implement the independent finite P/C/lock
formula. Every selected thread completes; actual CAS interleavings, accesses,
in-place outputs and all exterior cells are covered. Consumer readiness is an
explicit logical input condition, supplied by completed producers. -/
specification spinning_kernel_correctness (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) :
    Spec.Real (Logical.io h buf.cElements stage ⊨{Formula.Domain 0 h stage}
      stageFormula h buf.cElements stage) :=
  Logical.reader_implements h buf stage

#specNonCircular Formula.stage avoiding
  [runKernel,SourceReader.runPlan,Concurrent.Controller.run,Concurrent.execute,IO.io]
#specNonCircular Formula.Domain avoiding
  [runKernel,SourceReader.runPlan,Concurrent.Controller.run,
   Concurrent.EventProgramIO.ImplementsOn,Formula.stage]
#axiomsClean spinning_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Correct
