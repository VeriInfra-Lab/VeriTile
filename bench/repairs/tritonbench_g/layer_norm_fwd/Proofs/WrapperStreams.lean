import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperLaunchPrefix

/-! Per-call producer and current-stream contracts. The stream map is captured
for this actual wrapper invocation. The source never sets a per-device stream;
ordinary Torch device guards preserve that map. Ready means that this tensor's
producer precedes subsequently enqueued reads on the named stream. It is not
allocation freshness, a returned value equation, or a kernel-success predicate.
Actual source order ensures the transfer calls precede every dispatch read. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStream
open Wrapper WrapperNative
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

/-- Ordinary synchronized-call boundary: incoming tensors satisfy the caller's
usual tensor-library synchronization obligations. The producer contracts cover
work introduced by this invocation. They do not order unrelated external
producer streams or concurrent writes; resolve_neg may return its input. -/
structure API (tensor : TensorAPI world) (device : DispatchAPI world) where
  current : Nat → Nat
  ready : Tensor → Nat → Prop
  resolveStream : Role → Tensor → S → Nat
  resolve_stream : ∀ role input s, resolveStream role input s=current input.device
  resolve_producer : ∀ role input s output, ((tensor.resolve role input).run s).2=.ok output →
    ready output (resolveStream role input s)
  copySourceStream : Nat → Role → Tensor → S → Nat
  copyTargetStream : Nat → Role → Tensor → S → Nat
  copy_source : ∀ target role input s, copySourceStream target role input s=current input.device
  copy_target : ∀ target role input s, copyTargetStream target role input s=current target
  /-- Same-owner Tensor.to may alias. FIFO retains the preceding resolve write. -/
  same_owner_fifo : ∀ target role input s output, input.device=target →
    ready input (copySourceStream target role input s) →
    ((tensor.transfer target role input).run s).2=.ok output →
    ready output (copyTargetStream target role input s)
  /-- The explicit non_blocking=False transfer uses its own destination guard
  and orders the source producer before subsequent destination-stream reads. -/
  cross_owner_copy : ∀ target role input s output, input.device≠target →
    ready input (copySourceStream target role input s) →
    ((tensor.transfer target role input).run s).2=.ok output →
    ready output (copyTargetStream target role input s)
  allocationStream : Role → Allocation → S → Nat
  allocation_stream : ∀ role request s, allocationStream role request s=current request.device
  randint_producer : ∀ role request s output, request.randomSeed=Bool.true →
    ((tensor.allocate role request).run s).2=.ok output →
    ready output (allocationStream role request s)
  activeOwner : S → Nat
  activeStream : S → Nat
  /-- Actual Triton get_current_device/get_current_stream for this invocation. -/
  active_stream : ∀ s, activeStream s=current (activeOwner s)
  enter_owner : ∀ owner s, ((device.enter owner).run s).2=.ok () →
    activeOwner ((device.enter owner).run s).1=owner
  select_owner : ∀ grid s, activeOwner ((device.select grid).run s).1=activeOwner s
  admit_owner : ∀ args s, activeOwner ((device.admit args).run s).1=activeOwner s

def Local (api : API tensor device) (target : Nat) (value : Tensor) : Prop :=
  value.device=target ∧ api.ready value (api.current target)

def Optional (api : API tensor device) (target : Nat) : Option Tensor → Prop
  | none => True | some value => Local api target value

structure InputsReady (api : API tensor device) (target : Nat) (input : Inputs) : Prop where
  x : Local api target input.x
  weight : Local api target input.weight
  bias : Optional api target input.bias
  residual : Optional api target input.residual
  x1 : Optional api target input.x1
  weight1 : Optional api target input.weight1
  bias1 : Optional api target input.bias1
  rowscale : Optional api target input.rowscale

theorem materialize_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (target : Nat) (role : Role) (input : Tensor) (s : S) (output : Tensor)
    (h : ((materialize tensor target role input).run s).result=.ok output) : Local api target output := by
  unfold materialize at h
  dsimp only [Bind.bind] at h
  obtain ⟨resolved,hr,ht,_⟩ := bind_success _ _ _ _ h
  have hr' := call_success _ _ _ _ hr
  have ht' := call_success _ _ _ _ ht
  have howner := (tensor.resolve_metadata role input s resolved hr').2.2.1
  have hready := api.resolve_producer role input s resolved hr'
  rw [api.resolve_stream] at hready
  have hsource : api.ready resolved (api.copySourceStream target role resolved
      ((tensor.resolve role input).run s).1) := by
    simpa only [api.copy_source,howner] using hready
  have htarget : api.ready output (api.current target) := by
    by_cases he : resolved.device=target
    · simpa only [api.copy_target] using api.same_owner_fifo target role resolved _ output he hsource ht'
    · simpa only [api.copy_target] using api.cross_owner_copy target role resolved _ output he hsource ht'
  exact ⟨(tensor.transfer_metadata target role resolved _ output ht').2.2.1,htarget⟩

theorem optional_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (target : Nat) (role : Role) (input : Option Tensor) (s : S) (output : Option Tensor)
    (h : ((resolveOption tensor target role input).run s).result=.ok output) : Optional api target output := by
  cases input with
  | none =>
    have he : output=none := by simpa [resolveOption,Pure.pure,Action.pure] using h.symm
    simp [he,Optional]
  | some input =>
    unfold resolveOption at h
    dsimp only [Bind.bind] at h
    obtain ⟨value,hv,h,_⟩ := bind_success _ _ _ _ h
    have he : output=some value := by simpa [Pure.pure,Action.pure] using h.symm
    rw [he]
    exact materialize_ready tensor device api target role input s value hv

theorem inputs_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (q : Request) (s : S) (input : Inputs)
    (h : ((resolveInputs tensor q).run s).result=.ok input) : InputsReady api q.x.device input := by
  unfold resolveInputs at h
  dsimp only [Bind.bind] at h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨x,hx,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨w,hw,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨b,hb,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨r,hr,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨x1,hx1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨w1,hw1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨b1,hb1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨rs,hrs,h,_⟩ := bind_success _ _ _ _ h
  have he : input=⟨q.x,x,w,b,r,x1,w1,b1,rs⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [he]
  exact ⟨materialize_ready tensor device api q.x.device .x q.x _ x hx,
    materialize_ready tensor device api q.x.device .weight q.weight _ w hw,
    optional_ready tensor device api q.x.device .bias q.bias _ b hb,
    optional_ready tensor device api q.x.device .residual q.residual _ r hr,
    optional_ready tensor device api q.x.device .x1 q.x1 _ x1 hx1,
    optional_ready tensor device api q.x.device .weight1 q.weight1 _ w1 hw1,
    optional_ready tensor device api q.x.device .bias1 q.bias1 _ b1 hb1,
    optional_ready tensor device api q.x.device .rowscale q.rowscale _ rs hrs⟩

theorem optional_seed_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (condition : Bool) (request : Allocation) (s : S) (output : Option Tensor)
    (hrandom : request.randomSeed=Bool.true)
    (h : ((allocateOptional tensor .seeds condition request Bool.false).run s).result=.ok output) :
    Optional api request.device output := by
  cases condition
  · have he : output=none := by simpa [allocateOptional,Pure.pure,Action.pure] using h.symm
    simp [he,Optional]
  · unfold allocateOptional at h
    simp only [Bool.true_eq,ite_true,Bool.false_eq_true,ite_false,Bind.bind] at h
    obtain ⟨value,hv,h,_⟩ := bind_success _ _ _ _ h
    obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
    have he : output=some value := by simpa [Pure.pure,Action.pure] using h.symm
    rw [he]
    have hv' := call_success _ _ _ _ hv
    exact ⟨(tensor.allocate_metadata .seeds request s value hv').2.2.1,
      by simpa only [api.allocation_stream] using api.randint_producer .seeds request s value hrandom hv'⟩

theorem outputs_seed_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (q : Request) (v : Validated) (input : Inputs) (s : S) (output : Outputs)
    (h : ((allocateOutputs tensor q v input).run s).result=.ok output) :
    Optional api input.x.device output.seeds := by
  unfold allocateOutputs at h
  dsimp only [Bind.bind] at h
  obtain ⟨y,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨y1,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨r,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨mean,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨rstd,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨seeds,hs,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨mask,_,h,_⟩ := bind_success _ _ _ _ h
  have he : output=⟨y,y1,r,mean,rstd,seeds,mask⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [he]
  exact optional_seed_ready tensor device api q.dropout _ _ seeds rfl hs

theorem prepare_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare tensor q).run s).result=.ok p) :
    InputsReady api q.x.device p.inputs ∧ Optional api q.x.device p.outputs.seeds := by
  obtain ⟨_,_,v,input,output,block,_,hi,ho,_,he⟩ := prepare_success tensor q s p h
  have hinputs := inputs_ready tensor device api q _ input hi
  have hseeds := outputs_seed_ready tensor device api q v input _ output ho
  rw [hinputs.x.1] at hseeds
  rw [he]
  exact ⟨hinputs,hseeds⟩

#axiomsClean materialize_ready
#axiomsClean optional_ready
#axiomsClean inputs_ready
#axiomsClean optional_seed_ready
#axiomsClean outputs_seed_ready
#axiomsClean prepare_ready
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStream
