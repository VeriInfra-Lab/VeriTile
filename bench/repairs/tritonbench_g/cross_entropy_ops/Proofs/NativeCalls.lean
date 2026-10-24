import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Entries
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.InvocationCalls

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Observations Step groupSize groupRank lastStride fwdWidth bwdWidth warps snapshot_success LocalTrace RankTrace)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Successful ordinary calls with exact inputs, outputs, states, group and
work handle. Kernel cases use the source launch with its actual parameters.
The NCCL wait constructor contains the matching successful start; a mere
pending handle is insufficient. No constructor supplies numerical outputs. -/
inductive ActualCall {σ : Type} (rt : Runtime Expr σ) :
    CallKind → World Expr σ → World Expr σ → List Tensor → List Tensor → Nat → Option Nat → Prop
  | worldSize (group before count after) (success : rt.worldSize group before=.ok count after) :
      ActualCall rt .query before after [] [] group none
  | rank (group before rank after) (success : rt.rank group before=.ok rank after) :
      ActualCall rt .query before after [] [] group none
  | empty (shape dtype device before output after) (success : rt.empty shape dtype device before=.ok output after) :
      ActualCall rt .allocate before after [] [output] 0 none
  | emptyLike (input packed before output after) (success : rt.emptyLike input packed before=.ok output after) :
      ActualCall rt .allocate before after [input] [output] 0 none
  | enter (device before after) (success : rt.enterDevice device before=.ok () after) :
      ActualCall rt .context before after [] [] 0 none
  | exit (device before after) (success : rt.exitDevice device before=.ok () after) :
      ActualCall rt .context before after [] [] 0 none
  | setupForward (parameters warps before after) (success : rt.setupForwardOps parameters warps before=.ok () after) :
      ActualCall rt .setup before after [] [] 0 none
  | setupBackward (parameters warps before after) (success : rt.setupBackwardOps parameters warps before=.ok () after) :
      ActualCall rt .setup before after [] [] 0 none
  | save (logits lse labels original before after) (success : rt.saveTensors logits lse labels original before=.ok () after) :
      ActualCall rt .save before after [logits,lse,labels,original] [] 0 none
  | compare (input target before output after) (success : rt.equalInt input target before=.ok output after) :
      ActualCall rt .compare before after [input] [output] 0 none
  | forward (o logits labels losses lse zloss rows cols total start width split count before after)
      (success : forwardLaunch algebra rt logits.device
        (forwardParameters o logits labels losses lse zloss rows cols total start width split) count before=.ok () after) :
      ActualCall rt .forward before after [logits,labels] [losses,lse,zloss] 0 none
  | backward (c gradient upstream rows cols width count before after)
      (success : backwardLaunch algebra rt c.logits.device
        (backwardParameters c gradient upstream rows cols width) count before=.ok () after) :
      ActualCall rt .backward before after [c.logits,c.labels,c.lse,upstream] [gradient] 0 none
  | sum (input before output after) (success : rt.sum0 input before=.ok output after) :
      ActualCall rt .torchSum before after [input] [output] 0 none
  | lse (input before output after) (success : rt.logsumexp0 input before=.ok output after) :
      ActualCall rt .torchLSE before after [input] [output] 0 none
  | square (input before output after) (success : rt.square input before=.ok output after) :
      ActualCall rt .square before after [input] [output] 0 none
  | scale (scale input before output after) (success : rt.scalarMul scale input before=.ok output after) :
      ActualCall rt .scale before after [input] [output] 0 none
  | add (dst src before output after) (success : rt.addInplace dst src before=.ok output after) :
      ActualCall rt .add before after [dst,src] [output] 0 none
  | mask (dst mask before output after) (success : rt.maskedFillZero dst mask before=.ok output after) :
      ActualCall rt .mask before after [dst,mask] [output] 0 none
  | resolve (input before output after) (success : rt.resolveNeg input before=.ok output after) :
      ActualCall rt .resolve before after [input] [output] 0 none
  | clone (input before output after) (success : rt.cloneContiguous input before=.ok output after) :
      ActualCall rt .clone before after [input] [output] 0 none
  | contiguous (input before output after) (success : rt.contiguous input before=.ok output after) :
      ActualCall rt .contiguous before after [input] [output] 0 none
  | copy (dst src before output after) (success : rt.copy dst src before=.ok output after) :
      ActualCall rt .copy before after [dst,src] [output] 0 none
  | gather (dst src group before after) (success : rt.allGather dst src group before=.ok () after) :
      ActualCall rt .gather before after [src] [dst] group none
  | reduceStart (dst group before handle after)
      (started : rt.allReduceAsync dst group before=.ok handle after) :
      ActualCall rt .reduceStart before after [dst] [dst] group (some handle)
  | reduceWait (dst group start pending handle before after)
      (started : rt.allReduceAsync dst group start=.ok handle pending)
      (waited : rt.wait handle before=.ok () after) :
      ActualCall rt .reduceWait before after [dst] [dst] group (some handle)

  | zeros (input before output after) (success : rt.zerosLike input before=.ok output after) :
      ActualCall rt .allocate before after [input] [output] 0 none
  | mark (input before after) (success : rt.markNonDifferentiable input before=.ok () after) :
      ActualCall rt .save before after [input] [] 0 none

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
  ⟨call.runtime,.forward,call.data.zAllocated,call.data.computed,
   [call.data.workingLogits,call.data.labels],[call.data.losses,call.data.lse,call.data.zlosses],call.data.workingLogits.device,0,none,
   .kernel (forwardParameters call.options call.data.workingLogits call.data.labels call.data.losses call.data.lse call.data.zlosses
      call.data.rows call.data.cols (call.data.world*call.data.cols) (call.data.rank*call.data.cols) (fwdWidth call.data.cols)
      (decide (call.data.world>1 ∨ call.data.cols>65536))) call.data.count (warps (fwdWidth call.data.cols)),
   .forward _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ call.data.launchCall⟩

def backward {σ : Type} {rt : Runtime Expr σ} {c : Context} {input output before after}
    (b : BackwardData algebra rt c input output before after) : SourceCall σ :=
  ⟨rt,.backward,b.prepared,b.computed,[c.logits,c.labels,c.lse,b.upstream],[b.work],c.logits.device,0,none,
   .kernel (backwardParameters c b.work b.upstream b.rows b.cols (bwdWidth b.cols))
     (CrossEntropy1.Math.blocks b.cols (bwdWidth b.cols)) (warps (bwdWidth b.cols)),
   .backward _ _ _ _ _ _ _ _ _ b.launchCall⟩

def zeros {σ : Type} (rt : Runtime Expr σ) (input output : Tensor)
    (before after : World Expr σ) (success : rt.zerosLike input before=.ok output after) : SourceCall σ :=
  ⟨rt,.allocate,before,after,[input],[output],input.device,0,none,.plain,.zeros input before output after success⟩

def mark {σ : Type} (rt : Runtime Expr σ) (input : Tensor)
    (before after : World Expr σ) (success : rt.markNonDifferentiable input before=.ok () after) : SourceCall σ :=
  ⟨rt,.save,before,after,[input],[],input.device,0,none,.plain,.mark input before after success⟩

end SourceCall
#axiomsClean SourceCall.forward
#axiomsClean SourceCall.backward
#specNonCircular CallArguments avoiding [Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
