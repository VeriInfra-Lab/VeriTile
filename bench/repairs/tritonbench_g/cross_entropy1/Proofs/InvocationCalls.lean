import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ActualCalls

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Full nonnumerical arguments are part of the loaded ABI. The source kernel
parameters include strides, offsets, flags, grid and scalar constants. -/
inductive CallArguments (σ : Type) where
  | plain
  | natural (value : Nat)
  | integer (value : Int)
  | scalar (value : ℝ)
  | allocation (shape : List Nat) (dtype : String) (packed : Bool)
  | kernel (parameters : Parameters) (grid warps : Nat)
  | collective (dst gathered globalLSE : Tensor) (handle group : Nat)
      (start pending reduced waited : World Expr σ)

/-- An actual successful API boundary, including arguments discarded by the
legacy metadata-only registry. No arithmetic output equation is a field. -/
structure SourceCall (σ : Type) where
  runtime : Runtime Expr σ
  kind : CallKind
  before : World Expr σ
  after : World Expr σ
  inputs : List Tensor
  outputs : List Tensor
  device : Nat
  group : Nat
  work : Option Nat
  arguments : CallArguments σ
  actual : ActualCall runtime kind before after inputs outputs group work

namespace SourceCall

def sum {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.sum0 input before=.ok output after) : SourceCall σ :=
  ⟨rt,.torchSum,before,after,[input],[output],input.device,0,none,.plain,.sum input before output after success⟩

def lse {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.logsumexp0 input before=.ok output after) : SourceCall σ :=
  ⟨rt,.torchLSE,before,after,[input],[output],input.device,0,none,.plain,.lse input before output after success⟩

def square {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.square input before=.ok output after) : SourceCall σ :=
  ⟨rt,.square,before,after,[input],[output],input.device,0,none,.plain,.square input before output after success⟩

def scale {σ : Type} (rt : Runtime Expr σ) (scale : ℝ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.scalarMul scale input before=.ok output after) : SourceCall σ :=
  ⟨rt,.scale,before,after,[input],[output],input.device,0,none,.scalar scale,.scale scale input before output after success⟩

def add {σ : Type} (rt : Runtime Expr σ) (dst src output : Tensor)
    (before after : World Expr σ) (success : rt.addInplace dst src before=.ok output after) : SourceCall σ :=
  ⟨rt,.add,before,after,[dst,src],[output],dst.device,0,none,.plain,.add dst src before output after success⟩

def mask {σ : Type} (rt : Runtime Expr σ) (dst mask output : Tensor)
    (before after : World Expr σ) (success : rt.maskedFillZero dst mask before=.ok output after) : SourceCall σ :=
  ⟨rt,.mask,before,after,[dst,mask],[output],dst.device,0,none,.plain,.mask dst mask before output after success⟩

def resolve {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.resolveNeg input before=.ok output after) : SourceCall σ :=
  ⟨rt,.resolve,before,after,[input],[output],input.device,0,none,.plain,.resolve input before output after success⟩

def clone {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.cloneContiguous input before=.ok output after) : SourceCall σ :=
  ⟨rt,.clone,before,after,[input],[output],input.device,0,none,.plain,.clone input before output after success⟩

def contiguous {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.contiguous input before=.ok output after) : SourceCall σ :=
  ⟨rt,.contiguous,before,after,[input],[output],input.device,0,none,.plain,.contiguous input before output after success⟩

def copy {σ : Type} (rt : Runtime Expr σ) (dst src output : Tensor)
    (before after : World Expr σ) (success : rt.copy dst src before=.ok output after) : SourceCall σ :=
  ⟨rt,.copy,before,after,[dst,src],[output],dst.device,0,none,.plain,.copy dst src before output after success⟩

def compare {σ : Type} (rt : Runtime Expr σ) (input : Tensor) (target : Int) (output : Tensor)
    (before after : World Expr σ) (success : rt.equalInt input target before=.ok output after) : SourceCall σ :=
  ⟨rt,.compare,before,after,[input],[output],input.device,0,none,.integer target,.compare input target before output after success⟩

def empty {σ : Type} (rt : Runtime Expr σ) (shape : List Nat) (dtype : String) (device : Nat) (output : Tensor)
    (before after : World Expr σ) (success : rt.empty shape dtype device before=.ok output after) : SourceCall σ :=
  ⟨rt,.allocate,before,after,[],[output],device,0,none,.allocation shape dtype Bool.false,.empty shape dtype device before output after success⟩

def emptyLike {σ : Type} (rt : Runtime Expr σ) (input : Tensor) (packed : Bool) (output : Tensor)
    (before after : World Expr σ) (success : rt.emptyLike input packed before=.ok output after) : SourceCall σ :=
  ⟨rt,.allocate,before,after,[input],[output],input.device,0,none,.allocation input.shape input.dtype packed,.emptyLike input packed before output after success⟩

def forward {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra) : SourceCall σ :=
  ⟨call.runtime,.forward,call.data.lseAllocated,call.data.computed,
   [call.data.workingLogits,call.data.labels],[call.data.losses,call.data.lse],call.data.workingLogits.device,0,none,
   .kernel (forwardParameters call.options call.data.workingLogits call.data.labels call.data.losses call.data.lse
      call.data.rows call.data.cols (call.data.world*call.data.cols) (call.data.rank*call.data.cols) (fwdWidth call.data.cols)
      (decide (call.data.world>1 ∨ call.data.cols>65536))) call.data.count (warps (fwdWidth call.data.cols)),
   .forward _ _ _ _ _ _ _ _ _ _ _ _ _ _ call.data.launchCall⟩

def backward {σ : Type} {rt : Runtime Expr σ} {c : Context} {input output before after}
    (b : BackwardData algebra rt c input output before after) : SourceCall σ :=
  ⟨rt,.backward,b.prepared,b.computed,[c.logits,c.labels,c.lse,b.upstream],[b.work],c.logits.device,0,none,
   .kernel (backwardParameters c b.work b.upstream b.rows b.cols (bwdWidth b.cols))
     (Math.blocks b.cols (bwdWidth b.cols)) (warps (bwdWidth b.cols)),
   .backward _ _ _ _ _ _ _ _ _ b.launchCall⟩

end SourceCall

/-- Source slots are structural positions in the extracted successful trace,
not arbitrary successful calls on an admitted runtime. -/
def LocalMember {σ : Type} {rt : Runtime Expr σ} {count losses lse initial pair final}
    (_trace : LocalTrace rt count losses lse initial pair final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  count>1 ∧ ∃ (logged : World Expr σ)
    (hl : rt.logsumexp0 lse initial=.ok pair.2 logged)
    (hs : rt.sum0 losses logged=.ok pair.1 final),
      (slot=[0] ∧ c=SourceCall.lse rt lse pair.2 initial logged hl) ∨
      (slot=[1] ∧ c=SourceCall.sum rt losses pair.1 logged final hs)

def CorrectionMember {σ : Type} {rt : Runtime Expr σ} {scale losses lse initial output final}
    (_trace : CorrectionTrace rt scale losses lse initial output final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  scale≠0 ∧ ∃ (squared scaled : Tensor) (ss sm : World Expr σ)
    (hs : rt.square lse initial=.ok squared ss)
    (hm : rt.scalarMul scale squared ss=.ok scaled sm)
    (ha : rt.addInplace losses scaled sm=.ok output final),
      (slot=[0] ∧ c=SourceCall.square rt lse squared initial ss hs) ∨
      (slot=[1] ∧ c=SourceCall.scale rt scale squared scaled ss sm hm) ∨
      (slot=[2] ∧ c=SourceCall.add rt losses scaled output sm final ha)

/-- The wait record retains this exact start → pending → LSE → reduced → wait
chain. A reused numeric handle from another start does not produce this slot. -/
def RankMember {σ : Type} {rt : Runtime Expr σ} {o world rows pair initial output final}
    (trace : RankTrace rt o world rows pair initial output final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  match trace with
  | .direct _ => False
  | .reduced _ gathered globalLSE handle allocated collected pending reduced waited ha hg hs hl hw =>
      (slot=[0] ∧ c=SourceCall.empty rt [world,rows] pair.2.dtype pair.2.device gathered initial allocated ha) ∨
      (slot=[1] ∧ c=⟨rt,.gather,allocated,collected,[pair.2],[gathered],pair.2.device,o.processGroup.getD 0,none,
        .collective pair.1 gathered globalLSE handle (o.processGroup.getD 0) collected pending reduced waited,
        .gather _ _ _ _ _ hg⟩) ∨
      (slot=[2] ∧ c=⟨rt,.reduceStart,collected,pending,[pair.1],[pair.1],pair.1.device,o.processGroup.getD 0,some handle,
        .collective pair.1 gathered globalLSE handle (o.processGroup.getD 0) collected pending reduced waited,
        .reduceStart _ _ _ _ _ hs⟩) ∨
      (slot=[3] ∧ c=SourceCall.lse rt gathered globalLSE pending reduced hl) ∨
      (slot=[4] ∧ c=⟨rt,.reduceWait,reduced,waited,[pair.1],[pair.1],pair.1.device,o.processGroup.getD 0,some handle,
        .collective pair.1 gathered globalLSE handle (o.processGroup.getD 0) collected pending reduced waited,
        .reduceWait _ _ _ _ _ _ _ hs hw⟩)

def AggregateMember {σ : Type} {rt : Runtime Expr σ} {o world rows count labels losses lse initial pair final}
    (a : AggregateData rt o world rows count labels losses lse initial pair final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  (∃ rest, slot=0::rest ∧ LocalMember a.localCalls rest c) ∨
  (∃ rest, slot=1::rest ∧ RankMember a.rankCalls rest c) ∨
  (slot=[2] ∧ c=SourceCall.add rt a.rankPair.1 a.rankPair.2 a.addedLoss a.reduced a.added a.addCall) ∨
  (∃ rest, slot=3::rest ∧ CorrectionMember a.correctionCalls rest c) ∨
  (slot=[4] ∧ c=SourceCall.compare rt labels o.ignored a.mask a.corrected a.compared a.compareCall) ∨
  (slot=[5] ∧ c=SourceCall.mask rt a.correctedLoss a.mask pair.1 a.compared final a.maskCall)

def ForwardMember {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra)
    (slot : List Nat) (c : SourceCall σ) : Prop :=
  let d := call.data
  let snap := Classical.choice (snapshot_success call.runtime call.labels d.labels d.ranked d.labelsReady d.snapshotCall)
  (∃ (g : Nat) (hg : call.options.processGroup=some g), slot=[0] ∧
    c=⟨call.runtime,.query,call.initial,d.sized,[],[],call.logits.device,g,none,.natural d.world,
      .worldSize g call.initial d.world d.sized (by simpa only [groupSize,hg] using d.sizeCall)⟩) ∨
  (∃ (g : Nat) (hg : call.options.processGroup=some g), slot=[1] ∧
    c=⟨call.runtime,.query,d.sized,d.ranked,[],[],call.logits.device,g,none,.natural d.rank,
      .rank g d.sized d.rank d.ranked (by simpa only [groupRank,hg] using d.rankCall)⟩) ∨
  (slot=[2,0] ∧ c=SourceCall.resolve call.runtime call.labels snap.resolved d.ranked snap.intermediate snap.resolveCall) ∨
  (slot=[2,1] ∧ c=SourceCall.clone call.runtime snap.resolved d.labels snap.intermediate d.labelsReady snap.cloneCall) ∨
  (∃ h : lastStride call.logits≠1, slot=[3] ∧ c=SourceCall.contiguous call.runtime call.logits d.inplaceLogits
    d.labelsReady d.normalized (by simpa only [if_pos h] using d.originalNormalize)) ∨
  (slot=[4] ∧ c=SourceCall.resolve call.runtime d.inplaceLogits d.resolvedLogits d.normalized d.resolvedState d.resolveCall) ∨
  (∃ h : lastStride d.resolvedLogits≠1, slot=[5] ∧ c=SourceCall.contiguous call.runtime d.resolvedLogits d.workingLogits
    d.resolvedState d.resolved (by simpa only [if_pos h] using d.workingNormalize)) ∨
  (slot=[6] ∧ c=SourceCall.empty call.runtime (if d.count>1 then [d.count,d.rows] else [d.rows]) "torch.float32"
    d.workingLogits.device d.losses d.resolved d.lossAllocated d.lossCall) ∨
  (slot=[7] ∧ c=SourceCall.empty call.runtime (if d.count>1 then [d.count,d.rows] else [d.rows]) "torch.float32"
    d.workingLogits.device d.lse d.lossAllocated d.lseAllocated d.lseCall) ∨
  (slot=[8] ∧ c=SourceCall.forward call) ∨
  (∃ split rest, slot=9::rest ∧ AggregateMember (call.splitData split) rest c) ∨
  (slot=[10] ∧ c=⟨call.runtime,.save,d.aggregated,call.saved,[d.workingLogits,d.pair.2,d.labels,d.inplaceLogits],[],
    d.workingLogits.device,0,none,.plain,.save _ _ _ _ _ _ d.saveCall⟩)

def BackwardMember {σ : Type} {rt : Runtime Expr σ} {context input output before after}
    (b : BackwardData algebra rt context input output before after) (slot : List Nat) (c : SourceCall σ) : Prop :=
  let snap := Classical.choice (snapshot_success rt input b.upstream before b.upstreamReady b.snapshotCall)
  (slot=[0,0] ∧ c=SourceCall.resolve rt input snap.resolved before snap.intermediate snap.resolveCall) ∨
  (slot=[0,1] ∧ c=SourceCall.clone rt snap.resolved b.upstream snap.intermediate b.upstreamReady snap.cloneCall) ∨
  (∃ h : context.options.inplace=Bool.false, slot=[1] ∧ c=SourceCall.emptyLike rt context.logits Bool.false b.first
    b.upstreamReady b.allocated (by simpa only [h,Bool.false_eq_true,if_false] using b.firstCall)) ∨
  (∃ h : lastStride b.first≠1, slot=[2] ∧ c=SourceCall.emptyLike rt context.logits Bool.true b.work
    b.allocated b.prepared (by simpa only [if_pos h] using b.workCall)) ∨
  (slot=[3] ∧ c=SourceCall.backward b) ∨
  (∃ (different : b.work.objectId≠output.objectId) (returned : Tensor) (copy : rt.copy output b.work b.computed=.ok returned after),
    slot=[4] ∧ c=SourceCall.copy rt output b.work returned b.computed after copy)

/-- Participant is main forward, one peer forward, or main backward. The
slot plus participant is the work-instance identity, distinct from handles. -/
inductive Participant where
  | main | peer (rank : Nat) | backward
  deriving DecidableEq
structure SourceLocation where
  participant : Participant
  slot : List Nat
  deriving DecidableEq

#axiomsClean ForwardMember
#axiomsClean BackwardMember
#axiomsClean RankMember
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
