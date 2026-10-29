import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LocalRewrites
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Entries

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

inductive CallKind where
  | forward | backward | torchSum | torchLSE | square | scale | add | mask
  | resolve | clone | contiguous | copy | gather | reduceStart | reduceWait
  | query | allocate | context | setup | save | compare
  deriving DecidableEq

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
  | setupForward (parameters warps before after) (success : rt.setupForward parameters warps before=.ok () after) :
      ActualCall rt .setup before after [] [] 0 none
  | setupBackward (parameters warps before after) (success : rt.setupBackward parameters warps before=.ok () after) :
      ActualCall rt .setup before after [] [] 0 none
  | save (logits lse labels original before after) (success : rt.saveTensors logits lse labels original before=.ok () after) :
      ActualCall rt .save before after [logits,lse,labels,original] [] 0 none
  | compare (input target before output after) (success : rt.equalInt input target before=.ok output after) :
      ActualCall rt .compare before after [input] [output] 0 none
  | forward (o logits labels losses lse rows cols total start width split count before after)
      (success : forwardLaunch algebra rt logits.device
        (forwardParameters o logits labels losses lse rows cols total start width split) count before=.ok () after) :
      ActualCall rt .forward before after [logits,labels] [losses,lse] 0 none
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

/-- An object registry may identify an implementation only at a matching
ordinary API boundary. Digests and configuration are selection metadata, not
numerical ports. `loaded` is the ordinary loaded-object registry contract. -/
structure ActualObject {σ : Type} (allowed : Runtime Expr σ → Prop)
    (loaded : CallKind → String → String → List Nat → Prop) (descriptor : CallDescriptor σ) where
  runtime : Runtime Expr σ
  permitted : allowed runtime
  kind : CallKind
  before : World Expr σ
  after : World Expr σ
  beforeEq : descriptor.before=before.native
  afterEq : descriptor.after=after.native
  actual : ActualCall runtime kind before after descriptor.inputs descriptor.outputs descriptor.group descriptor.work
  object : loaded kind descriptor.implementation descriptor.objectDigest descriptor.configuration

def repairedSourceSHA256 : String :=
  "edfc7f0c6adb8c9a2050660c04d43034083fdefb58f068192208d91f676278d5"

/-- Selection is bound to actual API objects. Type-directed reduction
selection excludes, for example, attaching a Torch tree to a different API.
All supported collective algorithms map to the same actual matching wait;
the selected algorithm remains present in ReductionKind and the graph. -/
def reductionCallKind : ReductionKind → CallKind
  | .laneSum | .laneMaximum => .forward
  | .torchSum => .torchSum
  | .torchMaximum | .torchExponentialSum => .torchLSE
  | .collective _ => .reduceWait

structure PlatformCalls {σ : Type} (selection : Selection σ)
    (allowed : Runtime Expr σ → Prop) (loaded : CallKind → String → String → List Nat → Prop) where
  invocation : ∀ descriptor, selection.invoked descriptor → ActualObject allowed loaded descriptor
  reduction : ∀ site kind width tree, selection.reduction site kind width tree →
    ∃ h : selection.invoked site.call, (invocation site.call h).kind=reductionCallKind kind

/-- Every execution primitive is from an actual admitted call. This is an
instruction/object registry condition, separate from all numerical contracts. -/
structure TraceCalls {σ : Type} (selection : Selection σ) (trace : PrimitiveTrace σ) : Prop where
  raw : ∀ s x out, trace.raw s x out → selection.invoked s.call
  load : ∀ s state tensor index out, trace.load s state tensor index out →
    selection.invoked s.call ∧ tensor∈s.call.inputs
  literal : ∀ s x out, trace.literal s x out → selection.invoked s.call
  natural : ∀ s x out, trace.natural s x out → selection.invoked s.call
  negativeInfinity : ∀ s out, trace.negativeInfinity s out → selection.invoked s.call
  binary : ∀ s op a b out, trace.binary s op a b out → selection.invoked s.call
  unary : ∀ s op a out, trace.unary s op a out → selection.invoked s.call
  fma : ∀ s a b c out, trace.fma s a b c out → selection.invoked s.call
  hostScale : ∀ s scale a out, trace.hostScale s scale a out → selection.invoked s.call
  transfer : ∀ s edge a out, trace.transfer s edge a out → selection.invoked s.call
  store : ∀ s value state tensor index, trace.store s value state tensor index →
    selection.invoked s.call ∧ tensor∈s.call.outputs

/-- This metadata conclusion is derived from the actual store event. The
returned descriptor (including object id, region, shape, stride, offset and
lazy flags) is a member of the selected native call's exact output list. -/
theorem actual_store_object {σ : Type} {selection : Selection σ} {trace : PrimitiveTrace σ}
    {allowed loaded} (calls : PlatformCalls selection allowed loaded) (events : TraceCalls selection trace)
    {site value state tensor index} (store : trace.store site value state tensor index) :
    Nonempty (ActualObject allowed loaded site.call) ∧ tensor∈site.call.outputs := by
  obtain ⟨invoked,member⟩ := events.store _ _ _ _ _ store
  exact ⟨⟨calls.invocation _ invoked⟩,member⟩

theorem LocatedExecution.invoked {σ : Type} {selection : Selection σ} {trace : PrimitiveTrace σ}
    (links : TraceCalls selection trace) {readWord inputs} {graph : LocatedGraph σ width} {out : Nat}
    (run : LocatedExecution trace readWord inputs graph out) :
    ∀ site ∈ graph.sites, selection.invoked site.call := by
  induction run with
  | input => simp [LocatedGraph.sites]
  | loaded site state tensor index x out event binding =>
    simpa only [LocatedGraph.sites,List.mem_singleton,forall_eq] using (links.load _ _ _ _ _ event).1
  | raw site x out event => simpa only [LocatedGraph.sites,List.mem_singleton,forall_eq] using links.raw _ _ _ event
  | literal site x out event => simpa only [LocatedGraph.sites,List.mem_singleton,forall_eq] using links.literal _ _ _ event
  | natural site x out event => simpa only [LocatedGraph.sites,List.mem_singleton,forall_eq] using links.natural _ _ _ event
  | negativeInfinity site out event => simpa only [LocatedGraph.sites,List.mem_singleton,forall_eq] using links.negativeInfinity _ _ event
  | binary site op a b av bv out _ _ event ia ib =>
    intro found member
    simp only [LocatedGraph.sites,List.mem_cons,List.mem_append] at member
    rcases member with same | ha | hb
    · subst found; exact links.binary _ _ _ _ _ event
    · exact ia found ha
    · exact ib found hb
  | unary site op a av out _ event ih =>
    intro found member
    rcases List.mem_cons.mp member with same | inside
    · subst found; exact links.unary _ _ _ _ event
    · exact ih found inside
  | fma site a b c av bv cv out _ _ _ event ia ib ic =>
    intro found member
    simp only [LocatedGraph.sites,List.mem_cons,List.mem_append] at member
    rcases member with same | (ha | hb) | hc
    · subst found; exact links.fma _ _ _ _ _ event
    · exact ia found ha
    · exact ib found hb
    · exact ic found hc
  | hostScale site scale a av out _ event ih =>
    intro found member
    rcases List.mem_cons.mp member with same | inside
    · subst found; exact links.hostScale _ _ _ _ event
    · exact ih found inside
  | transfer site edge a av out _ event ih =>
    intro found member
    rcases List.mem_cons.mp member with same | inside
    · subst found; exact links.transfer _ _ _ _ event
    · exact ih found inside

theorem CompiledExecution.source_objects {σ : Type} {selection : Selection σ} {trace : PrimitiveTrace σ}
    {allowed loaded} (calls : PlatformCalls selection allowed loaded) (links : TraceCalls selection trace)
    {expression : Expr} {plan : SelectedPlan selection expression} {readWord}
    (run : CompiledExecution plan trace readWord) :
    ∀ site ∈ plan.compile.value.sites, Nonempty (ActualObject allowed loaded site.call) := by
  intro site member
  obtain ⟨actual,inside,same,_⟩ := run.origins site member
  have invoked := run.execution.invoked links actual inside
  rw [same] at invoked
  exact ⟨calls.invocation _ invoked⟩

#axiomsClean LocatedExecution.invoked
#axiomsClean CompiledExecution.source_objects
#axiomsClean actual_store_object
#specNonCircular PrimitiveContracts avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
