import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfileInstances
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LaunchContracts

namespace VeriTile.Bench.Repairs.CrossEntropy2.Invocation
open VeriTile.Triton Source Host Floating
open Floating.Profiled (DeviceFamily BackendABI BackwardABI BackwardOrder DeviceSelected TorchSelected BackwardSelected)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Auxiliary graph observations annotate the source result; no additional
runtime call or tensor operation is performed. -/
def mark {S A B : Type} (tag : A) (call : Step S B) : Step S (A × B) :=
  bind call fun value state => .ok (tag,value) state

theorem mark_success {S A B : Type} (tag actual : A) (call : Step S B) (value : B) (state : S) :
    mark tag call=.ok (actual,value) state ↔ tag=actual ∧ call=.ok value state := by
  cases call <;> simp [mark,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind,Prod.mk.injEq,and_assoc]

/-- Select the compiler graph only after this invocation's actual setup.
Same-width calls with different prepared pointers can select different graphs. -/
def forwardLaunch {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ)
    (p : Parameters) (count : Nat) (state : World Compiled σ) : Step (World Compiled σ) DeviceFamily :=
  if p.rows=0 then .ok Floating.Profiled.witnessDevice state else
  bind (rt.setupForward2 p count state) fun _ ready =>
    let device := abi.device (Floating.Scheduled.keyFor layout ready p)
    .ok device {ready with memory:=Grid.forward (Floating.Profiled.algebra device p.logitScale) p ready.memory count}

def hostFor {σ : Type} (layout : Layout σ) (abi : BackendABI)
    (lseState lossState : World Compiled σ) (lse loss : Tensor) : Profile :=
  {witnessProfile with
    hostMaxTree:=abi.torchLSEMax (Floating.Profiled.torchMetadata layout lseState lse)
    hostLSESumTree:=abi.torchLSESum (Floating.Profiled.torchMetadata layout lseState lse)
    hostSumTree:=abi.torchSum (Floating.Profiled.torchMetadata layout lossState loss)}

def finish {σ : Type} (rt : Runtime Compiled σ) (o : Options) (labels : Tensor)
    (pair : Tensor × Tensor) (reduced : World Compiled σ) : Step (World Compiled σ) Return :=
  bind (rt.addInplace pair.1 pair.2 reduced) fun losses added =>
  bind (rt.scalarMul o.squareScale pair.2 added) fun scaled ss =>
  bind (rt.multiply scaled pair.2 ss) fun zLoss multiplied =>
  bind (rt.addInplace losses zLoss multiplied) fun losses corrected =>
  bind (rt.toInt64 labels corrected) fun wideLabels widened =>
  bind (rt.equalInt wideLabels o.ignored widened) fun mask compared =>
  bind (rt.maskedFillZero losses mask compared) fun losses masked =>
  bind (rt.maskedFillZero zLoss mask masked) fun zLoss final => .ok (losses,pair.2,zLoss) final

/-- The three host plans are selected at the actual two reduction calls,
using each call's input and before-state. Their choices are invocation-local. -/
def aggregate {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ) (o : Options)
    (count : Nat) (labels loss lse : Tensor) (state : World Compiled σ) : Step (World Compiled σ) (Profile × Return) :=
  let localResult := if count>1 then
    bind (rt.logsumexp0 lse state) fun nextLSE logged =>
    bind (rt.sum0 loss logged) fun nextLoss summed =>
      .ok (hostFor layout abi state logged lse loss,(nextLoss,nextLSE)) summed
    else .ok (witnessProfile,(loss,lse)) state
  bind localResult fun chosen reduced => mark chosen.1 (finish rt o labels chosen.2 reduced)

def forward {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ) (o : Options)
    (logits labels : Tensor) (state : World Compiled σ) : Step (World Compiled σ) (DeviceFamily × Profile × Return) := by
  classical
  exact match logits.shape with
  | [rows,cols] =>
    if valid o logits labels rows cols then
      bind (normalize rt logits state) fun logits xs =>
      bind (normalize rt labels xs) fun labels ys =>
      let count := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width
      let split := o.split || decide (count>1 ∨ o.start≠0 ∨ cols≠o.total)
      let shape := if count>1 then [count,rows] else [rows]
      bind (rt.empty shape "torch.float32" logits.device ys) fun loss sl =>
      bind (rt.empty shape "torch.float32" logits.device sl) fun lse se =>
      bind (rt.empty shape "torch.float32" logits.device se) fun zLoss sz =>
      let p := parameters o logits labels loss lse zLoss rows cols split
      bind (forwardLaunch layout abi rt p count sz) fun device computed =>
      if split then mark device (aggregate layout abi rt o count labels loss lse computed)
      else .ok (device,witnessProfile,(loss,lse,zLoss)) computed
    else .error "invalid forward metadata" state
  | _ => .error "invalid logits rank" state

def backwardLaunch {σ : Type} (layout : Layout σ) (abi : BackwardABI) (device : DeviceFamily)
    (rt : Runtime Compiled σ) (p : Parameters) (count : Nat) (state : World Compiled σ) :
    Step (World Compiled σ) BackwardOrder :=
  if p.rows=0 then .ok ⟨fun _ => Bool.false,fun _ => Bool.false⟩ state else
  bind (rt.setupBackward2 p count state) fun _ ready =>
    .ok (abi (Floating.Profiled.backwardKey layout ready p))
      {ready with memory:=backwardGrid (Floating.Profiled.algebra device p.logitScale) p ready.memory count}

def backward {σ : Type} (layout : Layout σ) (abi : BackwardABI) (device : DeviceFamily)
    (rt : Runtime Compiled σ) (o : Options) (upstream logits lse labels : Tensor) (state : World Compiled σ) :
    Step (World Compiled σ) (BackwardOrder × Tensor) := by
  classical
  exact match logits.shape with
  | [rows,cols] =>
    if valid o logits labels rows cols ∧
        upstream.dtype="torch.float32" ∧ lse.dtype="torch.float32" ∧
        upstream.shape=[rows] ∧ lse.shape=[rows] ∧ upstream.device=logits.device ∧ lse.device=logits.device then
      bind (normalize rt logits state) fun logits xs =>
      bind (normalize rt labels xs) fun labels ys =>
      bind (normalize rt upstream ys) fun upstream gs =>
      bind (normalize rt lse gs) fun lse ls =>
      bind (rt.emptyLike logits Bool.true ls) fun gradient allocated =>
      let p := {parameters o logits labels upstream lse lse rows cols Bool.false with
        gradient:=gradient.region,gradientBase:=gradient.storageOffset,
        gradientStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride gradient,
        upstream:=upstream.region,upstreamBase:=upstream.storageOffset,
        upstreamStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride upstream}
      bind (backwardLaunch layout abi device rt p (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width) allocated)
        fun order computed => .ok (order,gradient) computed
    else .error "invalid backward metadata" state
  | _ => .error "invalid logits rank" state

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Invocation
