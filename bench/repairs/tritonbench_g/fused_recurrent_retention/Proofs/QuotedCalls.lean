import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedOwnership
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedStorage
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
noncomputable section
set_option maxHeartbeats 4000000

/-- Returned tensors retain the existing API metadata. Their numerical memory
comes solely from full quotation-grid execution and the existing host sum. -/
def forwardReturn [Inhabited α] (a : Arithmetic α) (p : PlanBinding s) (call : ForwardCall α s caller) :
    Option (ForwardReturn α) :=
  (forward a s call.args.buffers p call.allocation call.args.seed.isSome call.args.outputFinal call.before).map
    (fun memory => {
      memory := memory
      output := resultTensor (call.allocation.storage 2) call.args.q.view.device s.B s.H s.T s.DV
      finalState := if call.args.outputFinal then some (resultTensor (call.allocation.storage 1)
        call.args.q.view.device s.B s.H s.DK s.DV) else none
      context := call.args.saved })

def backwardReturn [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (call : BackwardCall α s ctx snapshot current caller) : Option (BackwardReturn α) :=
  (backward a s call.buffers p call.allocation ctx.initial.isSome
    (call.final.map TensorObject.view).isSome call.before).map (fun memory => {
      memory := memory
      dq := resultTensor (call.allocation.storage 3) ctx.q.view.device s.B s.H s.T s.DK
      dk := resultTensor (call.allocation.storage 4) ctx.q.view.device s.B s.H s.T s.DK
      dv := resultTensor (call.allocation.storage 5) ctx.q.view.device s.B s.H s.T s.DV
      initialGradient := none
      outputFlagGradient := none })

theorem forwardReturn_eq [Inhabited α] (a : Arithmetic α) (p : PlanBinding s) (call : ForwardCall α s caller) :
    forwardReturn a p call=some (call.sourceRun a p) := by
  unfold forwardReturn
  rw [forward_eq a s call.args.buffers p call.allocation call.args.seed.isSome call.args.outputFinal call.before
    ⟨call.scope.shape,call.scope.qDense,call.scope.vDense⟩
    (fun j hj => call.allocation.fresh j (call.inputs _ hj))]
  rfl

theorem backwardReturn_eq [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (call : BackwardCall α s ctx snapshot current caller) (domain : StoreDomain s call.buffers) :
    backwardReturn a p call=some (call.sourceRun a p) := by
  unfold backwardReturn
  rw [backward_eq a s call.buffers p call.allocation ctx.initial.isSome
    (call.final.map TensorObject.view).isSome call.before domain
    (fun j hj => call.allocation.fresh j (call.inputs _ hj))]
  rfl

/-- The wrapper performs only its already-existing assertions and packages
actual returned handles around the source-execution result. -/
def checkedForward [Inhabited α] (a : Arithmetic α) (p : PlanBinding s) (x : Host.Forward α s caller) :
    Option (Host.ForwardResult α) :=
  if ok : x.check=Bool.true then
    (forwardReturn a p (x.call ok)).map (fun out => ⟨out,
      Host.object (x.identities 2) (x.requiresGrad 2) out.output,
      out.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1))⟩)
  else none

def checkedBackward [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (x : Host.Backward α s ctx snapshot current caller) : Option (Host.BackwardResult α) :=
  if ok : x.check=Bool.true then
    (backwardReturn a p (x.call ok)).map (fun out => ⟨out,
      (Host.object (x.identities 3) (x.requiresGrad 3) out.dq,
       Host.object (x.identities 4) (x.requiresGrad 4) out.dk,
       Host.object (x.identities 5) (x.requiresGrad 5) out.dv,none,none)⟩)
  else none

theorem checkedForward_eq [Inhabited α] (a : Arithmetic α) (p : PlanBinding s) (x : Host.Forward α s caller) :
    checkedForward a p x=Host.forward a p x := by
  unfold checkedForward Host.forward
  split <;> simp only [forwardReturn_eq,Option.map_some]

theorem checkedBackward_eq [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (x : Host.Backward α s ctx snapshot current caller)
    (domain : StoreDomain s (ctx.buffers x.do_.view (x.final.map TensorObject.view))) :
    checkedBackward a p x=Host.backward a p x := by
  unfold checkedBackward Host.backward
  split
  · rename_i ok
    rw [backwardReturn_eq a p (x.call ok) domain]
    rfl
  · rfl

theorem savedDomain (call : ForwardCall α s caller) (d : Tensor4) (f : Option Tensor4) :
    StoreDomain s (call.args.saved.buffers d f) :=
  ⟨call.scope.shape,call.scope.qDense,call.scope.vDense⟩

def checkedForwardMemory [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (x : Host.Forward α s caller) : Memory α :=
  match checkedForward a p x with
  | some ret => ret.source.memory
  | none => x.before

def checkedBackwardMemory [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (x : Host.Backward α s ctx snapshot current caller) : Memory α :=
  match checkedBackward a p x with
  | some ret => ret.source.memory
  | none => x.before

theorem checkedForwardMemory_eq [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (x : Host.Forward α s caller) : checkedForwardMemory a p x=Host.forwardMemory a p x := by
  simp only [checkedForwardMemory,checkedForward_eq,Host.forwardMemory]
  cases Host.forward a p x <;> rfl

theorem checkedBackwardMemory_eq [Inhabited α] (a : Arithmetic α) (p : PlanBinding s)
    (x : Host.Backward α s ctx snapshot current caller)
    (domain : StoreDomain s (ctx.buffers x.do_.view (x.final.map TensorObject.view))) :
    checkedBackwardMemory a p x=Host.backwardMemory a p x := by
  simp only [checkedBackwardMemory,checkedBackward_eq a p x domain,Host.backwardMemory]
  cases Host.backward a p x <;> rfl

#specNonCircular forwardReturn avoiding [sourceForward,sourceBackward,resultO,resultQ,resultK,resultV,mathOutput,mathDQ,mathDK,mathDV]
#specNonCircular backwardReturn avoiding [sourceForward,sourceBackward,resultO,resultQ,resultK,resultV,mathOutput,mathDQ,mathDK,mathDV]
#axiomsClean forwardReturn_eq
#axiomsClean backwardReturn_eq
#axiomsClean checkedForward_eq
#axiomsClean checkedBackward_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
