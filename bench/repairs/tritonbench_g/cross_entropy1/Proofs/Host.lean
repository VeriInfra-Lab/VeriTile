import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardGrid
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorModel

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

structure World (α σ : Type) where
  memory : Memory α
  live : RegionName → Prop
  native : σ

inductive Step (S A : Type) where
  | error (detail : String) (state : S)
  | ok (value : A) (state : S)

def bind {S A B : Type} (result : Step S A) (next : A → S → Step S B) : Step S B :=
  match result with
  | .error e s => .error e s
  | .ok a s => next a s

def Ensures {S A : Type} (post : A → S → Prop) : Step S A → Prop
  | .error _ _ => True
  | .ok a s => post a s

theorem bind_ensures {S A B : Type} (r : Step S A) (f : A → S → Step S B)
    (p : A → S → Prop) (q : B → S → Prop) (first : Ensures p r)
    (rest : ∀ a s, p a s → Ensures q (f a s)) : Ensures q (bind r f) := by
  cases r with
  | error => trivial
  | ok a s => exact rest a s first

theorem self_ensures {S A : Type} (r : Step S A) : Ensures (fun a s => r=.ok a s) r := by
  cases r <;> trivial

theorem failed_prefix {S A B : Type} (e : String) (s : S) (f : A → S → Step S B) :
    bind (.error e s) f=.error e s := rfl

structure Options where
  smoothing : ℝ
  squareScale : ℝ
  ignored : Int
  inplace : Bool
  processGroup : Option Nat

structure Context where
  logits : Tensor
  lse : Tensor
  labels : Tensor
  inplaceLogits : Tensor
  options : Options
  totalClasses : Nat
  classStart : Nat

structure ForwardReturn where
  losses : Tensor
  context : Context

/-- Every effectful field is an ordinary local Torch/runtime operation. Kernel
execution is defined below and is deliberately absent from this interface. -/
structure Runtime (α σ : Type) where
  worldSize : Nat → World α σ → Step (World α σ) Nat
  rank : Nat → World α σ → Step (World α σ) Nat
  resolveNeg : Tensor → World α σ → Step (World α σ) Tensor
  cloneContiguous : Tensor → World α σ → Step (World α σ) Tensor
  contiguous : Tensor → World α σ → Step (World α σ) Tensor
  empty : List Nat → String → Nat → World α σ → Step (World α σ) Tensor
  emptyLike : Tensor → Bool → World α σ → Step (World α σ) Tensor
  enterDevice : Nat → World α σ → Step (World α σ) Unit
  setupForward : Parameters → Nat → World α σ → Step (World α σ) Unit
  setupBackward : Parameters → Nat → World α σ → Step (World α σ) Unit
  exitDevice : Nat → World α σ → Step (World α σ) Unit
  logsumexp0 : Tensor → World α σ → Step (World α σ) Tensor
  sum0 : Tensor → World α σ → Step (World α σ) Tensor
  allGather : Tensor → Tensor → Nat → World α σ → Step (World α σ) Unit
  allReduceAsync : Tensor → Nat → World α σ → Step (World α σ) Nat
  wait : Nat → World α σ → Step (World α σ) Unit
  addInplace : Tensor → Tensor → World α σ → Step (World α σ) Tensor
  square : Tensor → World α σ → Step (World α σ) Tensor
  scalarMul : ℝ → Tensor → World α σ → Step (World α σ) Tensor
  equalInt : Tensor → Int → World α σ → Step (World α σ) Tensor
  maskedFillZero : Tensor → Tensor → World α σ → Step (World α σ) Tensor
  saveTensors : Tensor → Tensor → Tensor → Tensor → World α σ → Step (World α σ) Unit
  copy : Tensor → Tensor → World α σ → Step (World α σ) Tensor

def rowStride (t : Tensor) : Nat := t.strides[0]?.getD 0
def lastStride (t : Tensor) : Nat := t.strides.getLast?.getD 0

def nextPower (n : Nat) : Nat := if n=0 then 0 else 2^(Nat.clog 2 n)
def fwdWidth (cols : Nat) : Nat := min (nextPower cols) 65536
def bwdWidth (cols : Nat) : Nat := min (nextPower cols) 4096

def warps (width : Nat) : Nat := if width<2048 then 4 else if width<8192 then 8 else 16

def forwardParameters (o : Options) (logits labels losses lse : Tensor)
    (rows cols total start width : Nat) (split : Bool) : Parameters :=
  {loss:=losses.region,lse:=lse.region,logits:=logits.region,gradient:=logits.region,
   upstream:=losses.region,labels:=Region.cast labels.region,
   lossBase:=losses.storageOffset,lseBase:=lse.storageOffset,logitsBase:=logits.storageOffset,
   labelsBase:=labels.storageOffset,rows:=rows,cols:=cols,totalClasses:=total,classStart:=start,
   ignored:=o.ignored,logitsStride:=rowStride logits,gradientStride:=rowStride logits,
   upstreamStride:=rowStride losses,block:=width,smoothing:=o.smoothing,squareScale:=o.squareScale,
   hasSmoothing:=decide (0<o.smoothing),split:=split}

def backwardParameters (c : Context) (gradient upstream : Tensor) (rows cols width : Nat) : Parameters :=
  {forwardParameters c.options c.logits c.labels upstream c.lse rows cols c.totalClasses c.classStart width Bool.false with
    gradient:=gradient.region,gradientBase:=gradient.storageOffset,gradientStride:=rowStride gradient,
    upstream:=upstream.region,upstreamBase:=upstream.storageOffset,upstreamStride:=rowStride upstream}

/-- Python context-manager cleanup also runs when launch setup fails. An
exit failure supersedes the body exception; otherwise the body exception and
its restored-device state propagate without running the caller continuation. -/
def withDevice {α σ A : Type} (rt : Runtime α σ) (device : Nat)
    (body : World α σ → Step (World α σ) A) (s : World α σ) : Step (World α σ) A :=
  bind (rt.enterDevice device s) fun _ entered =>
  match body entered with
  | .error reason failed =>
    match rt.exitDevice device failed with
    | .error exitReason closed => .error exitReason closed
    | .ok _ closed => .error reason closed
  | .ok value computed =>
    bind (rt.exitDevice device computed) fun _ closed => .ok value closed

def forwardLaunch {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (s : World α σ) : Step (World α σ) Unit :=
  withDevice rt device (fun entered =>
    bind (rt.setupForward p (warps p.block) entered) fun _ ready =>
      .ok () {ready with memory:=Grid.forward M p ready.memory count}) s

def backwardLaunch {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (s : World α σ) : Step (World α σ) Unit :=
  withDevice rt device (fun entered =>
    bind (rt.setupBackward p (warps p.block) entered) fun _ ready =>
      .ok () {ready with memory:=Grid.backward M p ready.memory count}) s

def snapshot {α σ : Type} (rt : Runtime α σ) (t : Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (rt.resolveNeg t s) fun resolved sr => rt.cloneContiguous resolved sr

/-- Exact host aggregation order: local column blocks first, rank collectives
second, wait before loss addition, optional square/multiply/add, final mask. -/
def aggregate {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows count : Nat)
    (labels losses lse : Tensor) (s : World α σ) : Step (World α σ) (Tensor × Tensor) :=
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
    bind (rt.scalarMul o.squareScale squared ss) fun scaled sm => rt.addInplace losses scaled sm
    else .ok losses added
  bind correction fun losses corrected =>
  bind (rt.equalInt labels o.ignored corrected) fun mask compared =>
  bind (rt.maskedFillZero losses mask compared) fun masked final => .ok (masked,pair.2) final

def groupSize {α σ : Type} (rt : Runtime α σ) (group : Option Nat) (s : World α σ) : Step (World α σ) Nat :=
  match group with
  | none => .ok 1 s
  | some g => rt.worldSize g s

def groupRank {α σ : Type} (rt : Runtime α σ) (group : Option Nat) (s : World α σ) : Step (World α σ) Nat :=
  match group with
  | none => .ok 0 s
  | some g => rt.rank g s

/-- Faithful selected wrapper, including guards, normalization, actual first
allocations, local/distributed aggregation and save-for-backward arguments. -/
def forward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (logits labels : Tensor) (s : World α σ) : Step (World α σ) ForwardReturn :=
  match logits.shape with
  | [rows,cols] =>
    if labels.shape=[rows] then
      bind (groupSize rt o.processGroup s) fun world sized =>
      bind (groupRank rt o.processGroup sized) fun rank ranked =>
      bind (snapshot rt labels ranked) fun labels labelsReady =>
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
        if width=0 then .error "integer division by zero" resolved else
          let count := Math.blocks cols width
          let split := decide (world>1 ∨ cols>65536)
          let lossShape := if count>1 then [count,rows] else [rows]
          bind (rt.empty lossShape "torch.float32" workingLogits.device resolved) fun losses lossAllocated =>
          bind (rt.empty lossShape "torch.float32" workingLogits.device lossAllocated) fun lse lseAllocated =>
          let p := forwardParameters o workingLogits labels losses lse rows cols (world*cols) (rank*cols) width split
          bind (forwardLaunch M rt workingLogits.device p count lseAllocated) fun _ computed =>
          let aggregation := if split then aggregate rt o world rows count labels losses lse computed
            else .ok (losses,lse) computed
          bind aggregation fun pair aggregated =>
          bind (rt.saveTensors workingLogits pair.2 labels inplaceLogits aggregated) fun _ saved =>
          .ok ⟨pair.1,⟨workingLogits,pair.2,labels,inplaceLogits,o,world*cols,rank*cols⟩⟩ saved
    else .error "labels shape assertion" s
  | _ => .error "logits rank unpacking" s

/-- Backward returns exactly the original saved in-place destination, or the
first actual empty_like result. A separate work allocation is copied back. -/
def backward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (upstream : Tensor) (s : World α σ) : Step (World α σ) Tensor :=
  bind (snapshot rt upstream s) fun upstream upstreamReady =>
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
      bind (backwardLaunch M rt c.logits.device p (Math.blocks cols width) prepared) fun _ computed =>
      if work.objectId≠output.objectId then
        bind (rt.copy output work computed) fun _ copied => .ok output copied
      else .ok output computed
  | _ => .error "logits rank unpacking" prepared

#axiomsClean forward
#axiomsClean backward
#axiomsClean failed_prefix
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
