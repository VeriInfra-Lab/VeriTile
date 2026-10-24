import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Source
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Execution
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Contract

namespace VeriTile.Bench.Repairs.L2NormTriton2.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormTriton1.Host (World Result Keep Dense Allocated Frame blockSize block_limit block_covers)
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- The unary runtime is shared by forward and context/allocation operations.
Backward adds requested-column normalization and its two-input launch setup. -/
structure Runtime (M : Type) where
  forward : L2NormTriton1.Host.Runtime M
  normalize : Nat → View → World M → Result M Dense
  setup : (p : Layout) → World M → Result M (L2NormTriton1.Host.Launch p.toLayout)

structure Call (α : Type) extends L2NormTriton1.Host.Call α where
  gradient : View

def Call.inputFor {α : Type} (call : Call α) (gradient : Bool) : View :=
  if gradient then call.gradient else call.input

def Call.cols {α : Type} (call : Call α) : Nat := call.gradient.shape.getLastD 0

structure BaseDomain {M α : Type} (call : L2NormTriton1.Host.Call α) (s : World M) : Prop where
  backed : call.input.backed
  live : call.input.storage∈s.allocations
  device : call.device=call.input.storage.owner

structure Domain {M α : Type} (call : Call α) (s : World M) : Prop where
  main : BaseDomain call.toCall s
  gradientBacked : call.gradient.backed
  gradientLive : call.gradient.storage∈s.allocations
  sameDevice : call.gradient.storage.owner=call.input.storage.owner
  direct : ∀ gradient,DirectView (call.inputFor gradient)
  positive : 0<call.cols

/-- Ordinary checked view(-1,N) contract. Both views keep the original storage
identity and exact input keys. N comes from DY; original shapes may differ. -/
structure Snapshot {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (N : Nat) (input : View) (before : World M) (d : Dense) (after : World M) : Prop where
  keep : Keep ops before after
  live : d.view.storage∈after.allocations
  storageObject : d.view.storage.object=input.storage.object
  device : d.view.storage.owner=input.storage.owner
  positive : 0<d.cols
  cols : d.cols=N
  count : d.count=input.numel
  keys : ∀ i,i<d.count → d.view.storage.key i=input.key i

theorem Snapshot.values {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {N : Nat} {input : View} {before after : World M} {d : Dense}
    (h : Snapshot ops neg N input before d after) (backed : input.backed)
    (live : input.storage∈before.allocations) (direct : DirectView input) (i : Nat) (hi : i<d.count) :
    ops.read after.memory (d.view.storage.key i)=logical ops neg before.memory input i := by
  rw [h.keys i hi,h.keep.reads _ live _ (backed i (by rw [←h.count];exact hi))]
  simp [logical,Negation.encode,direct.negative]

theorem Snapshot.aligned {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {N : Nat} {input : View} {before after : World M} {d : Dense}
    (h : Snapshot ops neg N input before d after) (direct : DirectView input)
    (nonempty : 0<d.count) : d.view.storage.baseByte%16=0 := by
  have key:=congrArg Key.address (h.keys 0 nonempty)
  simp only [Allocation.key,Nat.mul_zero,Nat.add_zero] at key
  rw [key]
  exact direct.aligned (by rw [←h.count];exact nonempty)

structure Library {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M) : Prop where
  shared : L2NormTriton1.Host.Library ops neg rt.forward
  normalize : ∀ N input s d after,rt.normalize N input s=.ok d after →
    input.backed → input.storage∈s.allocations → DirectView input → 0<N → Snapshot ops neg N input s d after
  normalizeFailure : ∀ N input s reason after,rt.normalize N input s=.error reason after → Keep ops s after
  setup : ∀ p s,Keep ops s (rt.setup p s).state

structure Prepared where
  snapshot : Dense
  gradient : Dense
  output : Dense
  block : Nat

def Prepared.inputFor (q : Prepared) (gradient : Bool) : Dense :=
  if gradient then q.gradient else q.snapshot

def Prepared.layout (q : Prepared) : Layout :=
  ⟨⟨q.snapshot.view.storage,q.output.view.storage,q.snapshot.rows,q.snapshot.cols,q.block⟩,
    q.gradient.view.storage⟩

/-- The retained shape/width branches are discharged by checked admission.
The two normalizations are direct views; allocation and row computation follow. -/
def prepare {M α : Type} (rt : Runtime M) (call : Call α) (s : World M) : Result M Prepared :=
  if call.gradient.shape=[] then .error "gradient last dimension missing" s else
  (rt.normalize call.cols call.input s).bind fun x s1 =>
  (rt.normalize call.cols call.gradient s1).bind fun dy s2 =>
  if dy.rows=x.rows ∧ dy.cols=x.cols then
    (rt.forward.empty x.view.storage.owner x.rows x.cols s2).bind fun y s3 =>
    if x.cols≤blockSize x.cols then .ok ⟨x,dy,y,blockSize x.cols⟩ s3
    else .error "feature dimension exceeds 64KB" s3
  else .error "reshaped matrices differ" s2

def launch {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (q : Prepared) (eps : γ) (s : World M) : Result M Unit :=
  let p := q.layout
  if p.rows=0 then .ok () s else
  (rt.setup p s).bind fun plan ready =>
  match plan with
  | .complete schedule _ => .ok () {ready with memory := Physical.run ops (a.compute p eps) p schedule ready.memory}
  | .interrupted completed storesDone _ _ reason =>
      .error reason {ready with memory := Physical.interrupted ops (a.compute p eps) p completed storesDone ready.memory}

/-- Shared context restoration executes after every body outcome. The returned
reshape refers to this call's output and uses the original X shape. -/
def execute {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ) (s : World M) : Result M View :=
  if call.selectedFP32 then
    (L2NormTriton1.Host.context rt.forward call.device (fun entered =>
      (prepare rt call entered).bind fun q ready =>
      (launch ops a rt q call.eps ready).bind fun _ done => .ok q.output done) s).bind fun y final =>
    match rt.forward.reshape y call.input.shape with
    | some out => .ok out final
    | none => .error "return reshape failed" final
  else .error "backward requires strided FP32 tensors on same CUDA device" s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Host
