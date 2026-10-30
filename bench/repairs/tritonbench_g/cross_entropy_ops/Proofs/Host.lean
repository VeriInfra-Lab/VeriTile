import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Grid
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Autograd

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Memory)
open CrossEntropy1.Host (World Step bind rowStride lastStride fwdWidth bwdWidth warps)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

structure Options extends CrossEntropy1.Host.Options where
  logitScale : ℝ

structure Context where
  logits : Tensor
  lse : Tensor
  labels : Tensor
  inplaceLogits : Tensor
  options : Options
  totalClasses : Nat
  classStart : Nat

def Context.base (c : Context) : CrossEntropy1.Host.Context :=
  ⟨c.logits,c.lse,c.labels,c.inplaceLogits,c.options.toOptions,c.totalClasses,c.classStart⟩

structure ForwardReturn where
  losses : Tensor
  zLosses : Tensor
  context : Context

/-- Reuse ordinary CE1 Torch/device/allocation operations. Ops has its own
kernel setup arguments, plus zeros_like and the nondifferentiable output mark.
The inherited CE1 setup callbacks are never called by this wrapper. -/
structure Runtime (α σ : Type) extends CrossEntropy1.Host.Runtime α σ where
  setupForwardOps : Parameters → Nat → World α σ → Step (World α σ) Unit
  setupBackwardOps : Parameters → Nat → World α σ → Step (World α σ) Unit
  zerosLike : Tensor → World α σ → Step (World α σ) Tensor
  markNonDifferentiable : Tensor → World α σ → Step (World α σ) Unit

def forwardParameters (o : Options) (logits labels losses lse zloss : Tensor)
    (rows cols total start width : Nat) (split : Bool) : Parameters :=
  {CrossEntropy1.Host.forwardParameters o.toOptions logits labels losses lse rows cols total start width split with
    zloss:=zloss.region,zlossBase:=zloss.storageOffset,logitScale:=o.logitScale}

def backwardParameters (c : Context) (gradient upstream : Tensor) (rows cols width : Nat) : Parameters :=
  {CrossEntropy1.Host.backwardParameters c.base gradient upstream rows cols width with
    zloss:=upstream.region,zlossBase:=upstream.storageOffset,logitScale:=c.options.logitScale}

def forwardLaunch {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (s : World α σ) : Step (World α σ) Unit :=
  CrossEntropy1.Host.withDevice rt.toRuntime device (fun entered =>
    bind (rt.setupForwardOps p (warps p.block) entered) fun _ ready =>
      .ok () {ready with memory:=Grid.forward M p ready.memory count}) s

def backwardLaunch {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (s : World α σ) : Step (World α σ) Unit :=
  CrossEntropy1.Host.withDevice rt.toRuntime device (fun entered =>
    bind (rt.setupBackwardOps p (warps p.block) entered) fun _ ready =>
      .ok () {ready with memory:=Grid.backward M p ready.memory count}) s

structure Aggregated where
  losses : Tensor
  lse : Tensor
  zLosses : Tensor

/-- Exact Python order, including the work window between all_reduce start and
wait, and the different split z-loss order `scale * square(lse)`. -/
def aggregate {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows count : Nat)
    (labels losses lse : Tensor) (s : World α σ) : Step (World α σ) Aggregated :=
  let localReduction := if count>1 then
    bind (rt.logsumexp0 lse s) fun nextLSE sl =>
    bind (rt.sum0 losses sl) fun nextLoss ss => .ok (nextLoss,nextLSE) ss
    else .ok (losses,lse) s
  bind localReduction fun localPair localState =>
  let rankReduction := if world>1 then
    bind (rt.empty [world,rows] localPair.2.dtype localPair.2.device localState) fun gathered allocated =>
    bind (rt.allGather gathered localPair.2 (o.processGroup.getD 0) allocated) fun _ collected =>
    bind (rt.allReduceAsync localPair.1 (o.processGroup.getD 0) collected) fun handle pending =>
    bind (rt.logsumexp0 gathered pending) fun globalLSE reduced =>
    bind (rt.wait handle reduced) fun _ waited => .ok (localPair.1,globalLSE) waited
    else .ok localPair localState
  bind rankReduction fun pair reduced =>
  bind (rt.addInplace pair.1 pair.2 reduced) fun losses added =>
  let correction := if o.squareScale≠0 then
    bind (rt.square pair.2 added) fun squared ss =>
    bind (rt.scalarMul o.squareScale squared ss) fun scaled sm =>
    bind (rt.equalInt labels o.ignored sm) fun mask compared =>
    bind (rt.maskedFillZero scaled mask compared) fun zlosses masked =>
    bind (rt.addInplace losses zlosses masked) fun corrected sc => .ok (corrected,zlosses) sc
    else bind (rt.zerosLike losses added) fun zeros sz => .ok (losses,zeros) sz
  bind correction fun corrected state =>
  bind (rt.equalInt labels o.ignored state) fun mask compared =>
  bind (rt.maskedFillZero corrected.1 mask compared) fun masked final =>
    .ok ⟨masked,pair.2,corrected.2⟩ final

/-- Full forward control flow. No callback is permitted to stand for the
wrapper's final output or desired mathematical result. -/
def forward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (logits labels : Tensor) (s : World α σ) : Step (World α σ) ForwardReturn :=
  match logits.shape with
  | [rows,cols] =>
    if labels.shape=[rows] then
      if cols=0 then .error "cross entropy requires a positive class count" s else
      bind (CrossEntropy1.Host.groupSize rt.toRuntime o.processGroup s) fun world sized =>
      bind (CrossEntropy1.Host.groupRank rt.toRuntime o.processGroup sized) fun rank ranked =>
      bind (CrossEntropy1.Host.snapshot rt.toRuntime labels ranked) fun labels labelsReady =>
      let normalize := if lastStride logits≠1 then rt.contiguous logits labelsReady else .ok logits labelsReady
      bind normalize fun inplaceLogits normalized =>
      if o.inplace && decide (rows>1 ∧ rowStride inplaceLogits<cols) then
        .error "inplace backward requires non-overlapping logit rows" normalized
      else
        bind (rt.resolveNeg inplaceLogits normalized) fun resolvedLogits resolvedState =>
        let normalizeWorking := if lastStride resolvedLogits≠1 then rt.contiguous resolvedLogits resolvedState
          else .ok resolvedLogits resolvedState
        bind normalizeWorking fun workingLogits resolved =>
        let width := fwdWidth cols
        let count := CrossEntropy1.Math.blocks cols width
        let split := decide (world>1 ∨ cols>65536)
        let lossShape := if count>1 then [count,rows] else [rows]
        bind (rt.empty lossShape "torch.float32" workingLogits.device resolved) fun losses lossAllocated =>
        bind (rt.empty lossShape "torch.float32" workingLogits.device lossAllocated) fun lse lseAllocated =>
        bind (rt.empty lossShape "torch.float32" workingLogits.device lseAllocated) fun zlosses zAllocated =>
        let p := forwardParameters o workingLogits labels losses lse zlosses rows cols (world*cols) (rank*cols) width split
        bind (forwardLaunch M rt workingLogits.device p count zAllocated) fun _ computed =>
        let aggregation := if split then aggregate rt o world rows count labels losses lse computed
          else .ok ⟨losses,lse,zlosses⟩ computed
        bind aggregation fun result aggregated =>
        bind (rt.saveTensors workingLogits result.lse labels inplaceLogits aggregated) fun _ saved =>
        bind (rt.markNonDifferentiable result.zLosses saved) fun _ marked =>
        .ok ⟨result.losses,result.zLosses,⟨workingLogits,result.lse,labels,inplaceLogits,o,world*cols,rank*cols⟩⟩ marked
    else .error "labels shape assertion" s
  | _ => .error "logits rank unpacking" s

/-- `grad_z_losses` is deleted before the exact four saved objects are used.
The Python adapter appends seven None values to this returned gradient. -/
def backward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (upstream : Tensor) (_zUpstream : Option Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (CrossEntropy1.Host.snapshot rt.toRuntime upstream s) fun upstream upstreamReady =>
  let first := if c.options.inplace then .ok c.logits upstreamReady else rt.emptyLike c.logits Bool.false upstreamReady
  bind first fun original allocated =>
  let output := if c.options.inplace then c.inplaceLogits else original
  let workspace := if lastStride original≠1 then rt.emptyLike c.logits Bool.true allocated else .ok original allocated
  bind workspace fun work prepared =>
  match c.logits.shape with
  | [rows,cols] =>
    let width := bwdWidth cols
    if width=0 then .error "integer division by zero" prepared else
      let p := backwardParameters c work upstream rows cols width
      bind (backwardLaunch M rt c.logits.device p (CrossEntropy1.Math.blocks cols width) prepared) fun _ computed =>
      if work.objectId≠output.objectId then
        bind (rt.copy output work computed) fun _ copied => .ok output copied
      else .ok output computed
  | _ => .error "logits rank unpacking" prepared

/-- The original eight-argument autograd ABI. -/
def backwardTuple (output : Tensor) : List (Option Tensor) :=
  [some output,none,none,none,none,none,none,none]

theorem backward_tuple_length (output : Tensor) : (backwardTuple output).length=8 := rfl

theorem logging_gradient_unused {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (upstream : Tensor) (z₁ z₂ : Option Tensor) (s : World α σ) :
    backward M rt c upstream z₁ s=backward M rt c upstream z₂ s := rfl

/-- Reuse the existing saved-object/version check without changing its four
objects. Logit scale is ordinary saved scalar metadata. -/
def backwardEntry {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (c : Context) (savedNative : σ)
    (upstream : Tensor) (zUpstream : Option Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (CrossEntropy1.Autograd.retrieve versions (CrossEntropy1.Autograd.capture versions c.base savedNative) s)
    (fun _ ready => backward M rt c upstream zUpstream ready)

theorem backward_entry_modified {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (c : Context) (savedNative : σ)
    (upstream : Tensor) (zUpstream : Option Tensor) (s : World α σ)
    (t : Tensor) (present : t∈CrossEntropy1.Autograd.savedTensors c.base)
    (changed : versions s.native t.objectId≠versions savedNative t.objectId) :
    backwardEntry M rt versions c savedNative upstream zUpstream s=.error "saved tensor version mismatch" s := by
  have mismatch : ¬CrossEntropy1.Autograd.Unmodified versions
      (CrossEntropy1.Autograd.capture versions c.base savedNative) s := fun h => changed (h t present)
  simp only [backwardEntry,CrossEntropy1.Autograd.retrieve,if_neg mismatch,CrossEntropy1.Host.bind]

theorem backward_entry_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (c : Context) (savedNative : σ)
    (upstream output : Tensor) (zUpstream : Option Tensor) (s after : World α σ)
    (success : backwardEntry M rt versions c savedNative upstream zUpstream s=.ok output after) :
    CrossEntropy1.Autograd.Unmodified versions (CrossEntropy1.Autograd.capture versions c.base savedNative) s ∧
    backward M rt c upstream zUpstream s=.ok output after := by
  by_cases same : CrossEntropy1.Autograd.Unmodified versions (CrossEntropy1.Autograd.capture versions c.base savedNative) s
  · exact ⟨same,by simpa only [backwardEntry,CrossEntropy1.Autograd.retrieve,if_pos same,CrossEntropy1.Host.bind] using success⟩
  · simp only [backwardEntry,CrossEntropy1.Autograd.retrieve,if_neg same,CrossEntropy1.Host.bind] at success
    cases success

#axiomsClean forward
#axiomsClean backward
#axiomsClean backward_tuple_length
#axiomsClean logging_gradient_unused
#axiomsClean backward_entry_modified
#axiomsClean backward_entry_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
