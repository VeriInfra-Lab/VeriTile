import bench.tritonbench_g.cross_entropy1.Proofs.Autograd
import bench.tritonbench_g.cross_entropy1.Proofs.PeerGraphs

namespace VeriTile.Bench.Repairs.CrossEntropy1.Entry
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- A successful call and its ordinary input/runtime contracts. All detailed
source, allocation, normalization and aggregation traces are extracted below.
There are no result-value, saved-LSE, or whole-output fields. -/
structure Invocation {α σ : Type} [Inhabited α] (obs : Observations α σ) (M : FP32 α) where
  runtime : Runtime α σ
  library : Library obs runtime
  groups : Groups obs runtime
  objects : ObjectRegistry runtime
  geometry : ViewGeometry runtime
  options : Options
  logits : Tensor
  labels : Tensor
  initial : World α σ
  saved : World α σ
  returned : ForwardReturn
  success : Host.forward M runtime options logits labels initial=.ok returned saved
  logitsLive : initial.live logits.region
  labelsLive : initial.live labels.region
  rank : logits.strides.length=logits.shape.length
  registered : objects.valid logits

def Invocation.data {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) : ForwardData M call.runtime call.options call.logits call.labels call.initial call.returned call.saved :=
  Classical.choice (forward_success M call.runtime call.options call.logits call.labels call.initial call.returned call.saved call.success)

def Invocation.kernel {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) : ForwardKernelData obs call.data :=
  Classical.choice (forward_kernel_data obs call.runtime call.library call.groups call.data call.logitsLive call.labelsLive call.rank)

def Invocation.views {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) : ForwardViewData obs call.objects call.data :=
  forward_view_data obs call.runtime call.library call.objects call.geometry call.kernel call.logitsLive call.rank call.registered

def Invocation.aggregation {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) : ForwardAggregation call.data := Classical.choice (forward_aggregation call.data)

def Invocation.splitData {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) (split : call.data.world>1 ∨ call.data.cols>65536) :
    AggregateData call.runtime call.options call.data.world call.data.rows call.data.count call.data.labels call.data.losses call.data.lse
      call.data.computed call.data.pair call.data.aggregated := by
  apply Classical.choice
  apply aggregate_success
  simpa only [decide_eq_true split,if_true] using call.data.aggregateCall

theorem Invocation.size {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) (many : call.data.world>1) :
    call.data.world=call.groups.size (call.options.processGroup.getD 0) := by
  have size := call.data.sizeCall
  cases group : call.options.processGroup with
  | none =>
    simp only [groupSize,group,Step.ok.injEq] at size
    omega
  | some g =>
    simp only [groupSize,group] at size
    simpa only [group,Option.getD_some] using (call.groups.sizeCall _ _ _ _ size).1

/-- Backward data is extracted only after actual saved-version retrieval.
The context is the exact context returned by the preceding forward call. -/
def Invocation.backwardData {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Invocation obs M) (versions : σ → Nat → Nat) (upstream output : Tensor)
    (before after : World α σ)
    (success : Autograd.backwardEntry M call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after) :
    BackwardData M call.runtime call.returned.context upstream output before after :=
  Classical.choice (Host.backward_success M call.runtime call.returned.context upstream output before after
    (Autograd.backward_success M call.runtime versions _ upstream output before after success).2)

def Invocation.realRun {σ : Type} {obs : Observations (WithBot ℝ) σ}
    (call : Invocation obs RealSource.algebra) (ops : TensorLibrary obs RealHost.arithmetic call.runtime)
    (split : call.data.world>1 ∨ call.data.cols>65536) : RealHost.RankRun σ obs :=
  ⟨call.runtime,call.library,ops,call.options,call.logits,call.labels,call.initial,call.saved,call.returned,
    call.data,call.kernel,call.splitData split,split⟩

def Invocation.fpPeer {σ : Type} {obs : Observations Floating.Expr σ}
    (call : Invocation obs Floating.algebra) (ops : TensorLibrary obs Floating.hostArithmetic call.runtime)
    (split : call.data.world>1 ∨ call.data.cols>65536) : Floating.Peer obs :=
  ⟨call.runtime,call.library,ops,call.options,call.logits,call.labels,call.initial,call.saved,call.returned,
    call.data,call.kernel,call.splitData split⟩

#axiomsClean Invocation.size
#axiomsClean Invocation.backwardData
#axiomsClean Invocation.realRun
#axiomsClean Invocation.fpPeer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Entry
