import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Correct
open VeriTile Triton FP.Structural ChunkDeltaFwd
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open scoped BigOperators
noncomputable section

/- Independent scalar mathematics and the complete logical output channels.
These definitions are definitionally connected to the canonical target used below. -/
namespace Mathematics
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
def corrected (xs : RealMath.Inputs) (K : Nat) (state : Nat → Nat → ℝ)
    (time column : Nat) : ℝ :=
  xs.value time column - ∑ e : Fin K, xs.delta time e.val * state e.val column

def state (xs : RealMath.Inputs) (T K stride width : Nat) (useSeed : Bool) :
    Nat → Nat → Nat → ℝ
  | 0,e,j => if useSeed then xs.seed e j else 0
  | c+1,e,j => state xs T K stride width useSeed c e j +
      ∑ t : Fin width, if c*stride+t.val < T then xs.key (c*stride+t.val) e *
        corrected xs K (state xs T K stride width useSeed c) (c*stride+t.val) j else 0
end Mathematics


/-- IO.lean fixes the quoted FP32 source, full CTA schedule, four logical input
channels, weight strides, optional seed/final masks and all three outputs. -/
def kernelIO (p : Parameters) (schedule : List PID) := LogicalTensors.io p schedule

def formula (p : Parameters) (xs : LogicalTensors.Inputs p) : LogicalTensors.Outputs p :=
  let state := fun head => Mathematics.state (LogicalTensors.realInputs p xs head)
    p.T p.K p.BT (((p.BT+p.BC-1)/p.BC)*p.BC) p.useSeed
  { history := fun head c k v => state head.val c.val k.val v.val
    corrected := fun head t v => Mathematics.corrected (LogicalTensors.realInputs p xs head.val) p.K
      (state head.val (t.val/p.BT)) t.val v.val
    final := fun head k v => state head.val p.NT k.val v.val }

/-- Complete kernel grid: independent rounded-window recurrence, actual source
access safety, every history/corrected/final value and the exterior cell frame. -/
specification chunk_delta_kernel_correct (p : Parameters) (shape : Geometry.ShapeValid p)
    (schedule : List PID) (order : schedule.Perm (grid p)) :
    Spec.Real (kernelIO p schedule ⊨ formula p) :=
  LogicalTensors.correct_recurrence p schedule shape order Mathematics.state (by intros; rfl) (by intros; rfl)


#stmtSurfaceSubset chunk_delta_kernel_correct ⊆
  [Parameters,Geometry.ShapeValid,PID,List.Perm,grid,Spec.Real,LogicalProgramIO.Implements,kernelIO,formula,LogicalTensors.Inputs,LogicalTensors.Outputs]
#specNonCircular formula avoiding [ChunkDeltaFwd.kernel,FP.Structural.exec,runGrid,gridMemory]
#axiomsClean chunk_delta_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.Correct
