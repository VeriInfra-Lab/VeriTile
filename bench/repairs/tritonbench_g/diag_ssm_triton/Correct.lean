import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.LogicalIO
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ReaderConnections

/-! Four per-program kernel specifications, each covering the entire time loop.
The shared regional IO relation carries actual source access safety, successful
execution, all active typed outputs and the exact raw-cell frame. IO.lean gives
the source identity, finite input/output windows and independent recurrences.
Read-only aliases are admitted; the existing output-separation premises stay
visible. Only this mathematical layer erases source storage precision. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Correctness
open VeriTile VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
open Interface
noncomputable section

/-- Independent forward recurrence, with state t+1 stored at time t. -/
def state {α : Type} [Semiring α] (initial transition : α) (input : Nat → α) : Nat → α
  | 0 => initial
  | t+1 => state initial transition input t * transition + input t

/-- Reverse carried adjoint; complex callers pass the conjugated transition. -/
def carry {α : Type} [Semiring α] (length : Nat) (transition : α) (gy : Nat → α) : Nat → α
  | 0 => 0
  | k+1 => (gy (length-1-k)+carry length transition gy k)*transition

/-- Per-batch transition gradient: previous states are independent saved inputs.
There is no later host batch reduction in this target. -/
def transitionSum {α : Type} [Semiring α] (length : Nat) (transition initial : α)
    (saved gy : Nat → α) : Nat → α
  | 0 => 0
  | k+1 => transitionSum length transition initial saved gy k +
      (gy (length-1-k)+carry length transition gy k) *
        (if length-1-k=0 then initial else saved (length-1-k-1))

def forwardMath {α : Type} [Semiring α] {B length : Nat}
    (xs : LogicalIO.ForwardInputs α B length) (time : Fin length) (lane : Fin B) : α :=
  state (xs.initial lane) (xs.transition lane) (fun t => LogicalIO.timeAt xs.input t lane) (time.val+1)

/-- All time/lane input gradients and both terminal per-lane gradients. -/
def backwardMath {α : Type} [Semiring α] (conjugate : α → α) {B length : Nat}
    (xs : LogicalIO.BackwardInputs α B length) : LogicalIO.BackwardOutputs α B length :=
  ⟨(fun time lane =>
      let transition := conjugate (xs.transition lane)
      let gy := fun t => LogicalIO.timeAt xs.upstream t lane
      gy time.val+carry length transition gy (length-1-time.val)),
    (fun lane => carry length (conjugate (xs.transition lane)) (fun t => LogicalIO.timeAt xs.upstream t lane) length),
    (fun lane => transitionSum length (conjugate (xs.transition lane)) (conjugate (xs.initial lane))
      (fun t => conjugate (LogicalIO.timeAt xs.saved t lane)) (fun t => LogicalIO.timeAt xs.upstream t lane) length)⟩

/-- Complete FP32 Real forward time scan at every active logical lane. -/
specification forward_region_correctness (p : ForwardLayout) (length : Nat) (sep : p.x≠p.y) :
    Spec.Real (LogicalIO.forwardRealView p length ⊨ forwardMath) :=
  ReaderConnections.forwardReal p length sep state (fun _ _ _ => rfl) (fun _ _ _ _ => rfl)
#axiomsClean forward_region_correctness
#stmtSurfaceSubset forward_region_correctness ⊆
  [Spec.Real,VeriTile.Triton.LogicalIO.Implements,LogicalIO.forwardRealView,forwardMath,
   LogicalIO.ForwardInputs,LogicalIO.BackwardOutputs,
   ForwardLayout,ForwardLayout.B,ForwardLayout.x,ForwardLayout.y,RegionName,
   RegionCorrectness.BackwardSeparation,ComplexBackwardRun.Separation,
   forwardIO,backwardIO,UKernelIO.Ctx,UKernelIO.nOut,UKernelIO.oarity,UKernelIO.oty,ChanTy.carrier,UKernelIO.RegionalImplements]

/-- Complete FP32 Real backward time scan at every active logical lane. -/
specification backward_region_correctness (p : BackwardLayout) (length : Nat) (sep : RegionCorrectness.BackwardSeparation p) :
    Spec.Real (LogicalIO.backwardRealView p length ⊨ backwardMath id) :=
  ReaderConnections.backwardReal p length sep carry transitionSum
    (fun _ _ _ => rfl) (fun _ _ _ _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ _ _ _ _ _ => rfl)
#axiomsClean backward_region_correctness
#stmtSurfaceSubset backward_region_correctness ⊆
  [Spec.Real,VeriTile.Triton.LogicalIO.Implements,LogicalIO.backwardRealView,backwardMath,
   LogicalIO.BackwardInputs,LogicalIO.BackwardOutputs,
   BackwardLayout,BackwardLayout.B,ForwardLayout.x,ForwardLayout.y,RegionName,
   RegionCorrectness.BackwardSeparation,ComplexBackwardRun.Separation,
   forwardIO,backwardIO,UKernelIO.Ctx,UKernelIO.nOut,UKernelIO.oarity,UKernelIO.oty,ChanTy.carrier,UKernelIO.RegionalImplements]

/-- Complete complex64 forward time scan at every active logical lane. -/
specification complex_forward_region_correctness (p : ForwardLayout) (length : Nat) :
    Spec.Real (LogicalIO.forwardComplexView p length ⊨ forwardMath) :=
  ReaderConnections.forwardComplex p length state (fun _ _ _ => rfl) (fun _ _ _ _ => rfl)
#axiomsClean complex_forward_region_correctness
#stmtSurfaceSubset complex_forward_region_correctness ⊆
  [Spec.Real,VeriTile.Triton.LogicalIO.Implements,LogicalIO.forwardComplexView,forwardMath,
   LogicalIO.ForwardInputs,LogicalIO.BackwardOutputs,
   ForwardLayout,ForwardLayout.B,ForwardLayout.x,ForwardLayout.y,RegionName,
   RegionCorrectness.BackwardSeparation,ComplexBackwardRun.Separation,
   forwardIO,backwardIO,UKernelIO.Ctx,UKernelIO.nOut,UKernelIO.oarity,UKernelIO.oty,ChanTy.carrier,UKernelIO.RegionalImplements]

/-- Complete complex64 backward time scan at every active logical lane. -/
specification complex_backward_region_correctness (p : BackwardLayout) (length : Nat) (sep : ComplexBackwardRun.Separation p) :
    Spec.Real (LogicalIO.backwardComplexView p length ⊨ backwardMath (starRingEnd ℂ)) :=
  ReaderConnections.backwardComplex p length sep carry transitionSum
    (fun _ _ _ => rfl) (fun _ _ _ _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ _ _ _ _ _ => rfl)
#axiomsClean complex_backward_region_correctness
#stmtSurfaceSubset complex_backward_region_correctness ⊆
  [Spec.Real,VeriTile.Triton.LogicalIO.Implements,LogicalIO.backwardComplexView,backwardMath,
   LogicalIO.BackwardInputs,LogicalIO.BackwardOutputs,
   BackwardLayout,BackwardLayout.B,ForwardLayout.x,ForwardLayout.y,RegionName,
   RegionCorrectness.BackwardSeparation,ComplexBackwardRun.Separation,
   forwardIO,backwardIO,UKernelIO.Ctx,UKernelIO.nOut,UKernelIO.oarity,UKernelIO.oty,ChanTy.carrier,UKernelIO.RegionalImplements]

#specNonCircular forwardMath avoiding [forwardKernel,backwardKernel,forwardIO,backwardIO]
#specNonCircular backwardMath avoiding [forwardKernel,backwardKernel,forwardIO,backwardIO]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.Correctness
