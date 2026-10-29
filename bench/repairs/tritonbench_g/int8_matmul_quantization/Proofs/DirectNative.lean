import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectRow

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 1500000

/-- Compiler/source structure without caller-assumed launch or index bounds. -/
structure GemmSource (I F H : Type) (s : Shape) (b : Buffers) where
  invocation : Invocation s
  parsedProduction : String
  sourceIdentity : parsedProduction=Source.productionAST
  admitted : Source.Admitted invocation.config
  dispatch : invocation.schedule.events.Perm (allCommits s invocation.config)
  lowered : Code I F H Unit
  sourceLowering : lowered=gridCode s invocation.config b invocation.selected invocation.schedule.events
def GemmSource.bind (call : GemmSource I F H s b) (bounds : IndexABI s call.invocation.config b) : BoundCall I F H s b :=
  ⟨call.invocation,call.parsedProduction,call.sourceIdentity,call.admitted,bounds,call.dispatch,call.lowered,call.sourceLowering⟩

structure RowSource (I F H : Type) (G K : Nat) (q : QuantBuffers) where
  invocation : QuantInvocation G K
  parsedProduction : String
  sourceIdentity : parsedProduction=Source.productionAST
  configIndex : Fin 4
  lowered : Code I F H Unit
  sourceLowering : lowered=quantGridCode q invocation.plan invocation.groups
def RowSource.bind (call : RowSource I F H G K q) (bounds : QuantIndexABI G K (width K) q)
    (rowStructure : q.transposeOutput=Bool.false ∧ q.scalePrecision=.half ∧ q.initialMaximumZero=Bool.true) : BoundQuantCall I F H G K q :=
  ⟨call.invocation,call.parsedProduction,call.sourceIdentity,call.configIndex,width K,bounds,rowStructure,call.lowered,call.sourceLowering⟩

def gemmBound (r : GemmRequest) (out : Nat) (call : GemmSource I F H r.shape (r.buffers out))
    (active : ¬(r.shape.M=0 ∨ r.shape.N=0)) : BoundCall I F H r.shape (r.buffers out) :=
  call.bind (nativeGemm r.a.info (some r.aScale.info) r.b.info r.bScale.info (r.out.map TensorView.info)
    Bool.false 0 0 out r.contract (by change ¬(rows r.a.info=0 ∨ cols r.b.info=0) at active;omega)
    (by change ¬(rows r.a.info=0 ∨ cols r.b.info=0) at active;omega) call.invocation.config call.admitted)
def rowBound (r : RowRequest) (data scales : Nat) (call : RowSource I F H r.G r.K (r.buffers data scales))
    (active : r.G≠0) : BoundQuantCall I F H r.G r.K (r.buffers data scales) :=
  call.bind (nativeRow r.input.info data scales r.contract (by change rows r.input.info≠0 at active;omega)) ⟨rfl,rfl,rfl⟩

def gemmNative (r : GemmRequest) (out : Nat) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (callerDevice : Nat) (trials : List (GemmSource I F H r.shape (r.buffers out)))
    (chosen : GemmSource I F H r.shape (r.buffers out)) (state : σ) : NativeReturned σ :=
  if empty : r.shape.M=0 ∨ r.shape.N=0 then
    ⟨outputTensor r.shape (r.buffers out) r.owner,hook.zero r.shape (r.buffers out) state,callerDevice⟩
  else compiledWrapper abi hook r.owner callerDevice (trials.map (fun c=>gemmBound r out c empty)) (gemmBound r out chosen empty) state
theorem gemmNative_refines (r : GemmRequest) (out : Nat) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (callerDevice : Nat) (trials : List (GemmSource I F H r.shape (r.buffers out)))
    (chosen : GemmSource I F H r.shape (r.buffers out)) (state : σ) :
    let result:=gemmNative r out abi hook callerDevice trials chosen state
    let reference:=wrapper a r.shape (r.buffers out) r.owner callerDevice (trials.map GemmSource.invocation) chosen.invocation (abi.observe state)
    (result.tensor,abi.observe result.state,result.currentDevice)=(reference.tensor,reference.memory,reference.currentDevice) := by
  unfold gemmNative
  split_ifs with empty
  · simp only [wrapper,if_pos empty,hook.law]
  · simpa only [List.map_map] using compiledWrapper_refines abi hook r.owner callerDevice
      (trials.map (fun c=>gemmBound r out c empty)) (gemmBound r out chosen empty) state

theorem quantRun_empty (a : Ops I F H) (q : QuantBuffers) (chosen : QuantInvocation 0 K) (m : Memory F H) :
    quantRun a q chosen m=m := by
  have groups : chosen.groups=[]:=by simpa using chosen.complete.eq_nil
  simp [quantRun,quantGridCode,groups,pure,eval_pure]
theorem quantWrapper_empty (a : Ops I F H) (q : QuantBuffers) (trials : List (QuantInvocation G K))
    (chosen : QuantInvocation G K) (m : Memory F H) (empty : G=0) : quantWrapper a q trials chosen m=m := by
  subst G
  have trial : ∀ trials : List (QuantInvocation 0 K),quantTrials a q trials m=m := by
    intro trials;induction trials with
    | nil=>rfl
    | cons c cs ih=>simpa [quantTrials,quantRun_empty] using ih
  rw [quantWrapper,quantRun_empty,trial]
def rowNative (r : RowRequest) (data scales : Nat) (abi : LocalEventABI I F H σ a)
    (trials : List (RowSource I F H r.G r.K (r.buffers data scales)))
    (chosen : RowSource I F H r.G r.K (r.buffers data scales)) (state : σ) : σ :=
  if empty : r.G=0 then state else compiledQuantWrapper abi
    (trials.map (fun c=>rowBound r data scales c empty)) (rowBound r data scales chosen empty) state
theorem rowNative_refines (r : RowRequest) (data scales : Nat) (abi : LocalEventABI I F H σ a)
    (trials : List (RowSource I F H r.G r.K (r.buffers data scales)))
    (chosen : RowSource I F H r.G r.K (r.buffers data scales)) (state : σ) :
    abi.observe (rowNative r data scales abi trials chosen state)=
      quantWrapper a (r.buffers data scales) (trials.map RowSource.invocation) chosen.invocation (abi.observe state) := by
  unfold rowNative
  split_ifs with empty
  · exact (quantWrapper_empty a _ _ _ _ empty).symm
  · simpa only [List.map_map] using compiledQuantWrapper_refines abi
      (trials.map (fun c=>rowBound r data scales c empty)) (rowBound r data scales chosen empty) state

structure NativeGemmAllocation (r : GemmRequest) (abi : LocalEventABI I F H σ a) (caller : σ) (live : Nat → Prop) where
  output : TensorView
  state : σ
  descriptor : GemmDescriptor r output
  fresh : r.out=none → ¬live output.info.storage
  frame : ∀ p,p.1≠output.info.storage ∨ r.out.isSome=Bool.true → abi.observe state p=abi.observe caller p
  outputIndex : ∀ i,i<r.shape.M*r.shape.N → output.physicalIndex i=offset (r.out.map TensorView.info)+i
def NativeGemmAllocation.logical (alloc : NativeGemmAllocation r abi caller live) : GemmAllocation r (abi.observe caller) live :=
  ⟨alloc.output,abi.observe alloc.state,alloc.descriptor,alloc.fresh,alloc.frame,alloc.outputIndex⟩
structure NativeHostResult (σ : Type) where
  tensor : TensorView
  state : σ
  currentDevice : Nat
def runGemmNative (r : GemmRequest) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (callerDevice : Nat) (caller : σ) (live : Nat → Prop) (alloc : NativeGemmAllocation r abi caller live)
    (trials : List (GemmSource I F H r.shape (r.buffers alloc.output.info.storage)))
    (chosen : GemmSource I F H r.shape (r.buffers alloc.output.info.storage)) : Except InputError (NativeHostResult σ) := do
  checkGemmRequest r
  let done:=gemmNative r alloc.output.info.storage abi hook callerDevice trials chosen alloc.state
  return ⟨alloc.output,done.state,done.currentDevice⟩

theorem gemm_native_complete (r : GemmRequest) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (callerDevice : Nat) (caller : σ) (live : Nat → Prop) (inputs : r.live live) (alloc : NativeGemmAllocation r abi caller live)
    (trials : List (GemmSource I F H r.shape (r.buffers alloc.output.info.storage)))
    (chosen : GemmSource I F H r.shape (r.buffers alloc.output.info.storage)) :
    ∃ result,runGemmNative r abi hook callerDevice caller live alloc trials chosen=.ok result ∧
      result.tensor=alloc.output ∧ GemmDescriptor r result.tensor ∧ OutputViewBinding r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,abi.observe result.state (outputAddress r.shape (r.buffers result.tensor.info.storage) cell)=
        .inr (wordReference a r.shape chosen.invocation.config (r.buffers result.tensor.info.storage) chosen.invocation.selected chosen.invocation.schedule (abi.observe caller) cell)) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape (r.buffers result.tensor.info.storage) p → abi.observe result.state p=abi.observe caller p) := by
  have ref:=gemmNative_refines r alloc.output.info.storage abi hook callerDevice trials chosen alloc.state
  have mem:=congrArg (fun t=>t.2.1) ref
  have dev:=congrArg (fun t=>t.2.2) ref
  simp only [Prod.fst,Prod.snd] at mem dev
  have sep : (r.buffers alloc.output.info.storage).Separate:=alloc.logical.separate r _ live inputs
  let done:=gemmNative r alloc.output.info.storage abi hook callerDevice trials chosen alloc.state
  refine ⟨⟨alloc.output,done.state,done.currentDevice⟩,?_,rfl,alloc.descriptor,alloc.logical.binding,?_,?_,?_⟩
  · simp [runGemmNative,checkGemmRequest_ok,Except.bind,Except.pure,bind,pure,done]
  · exact dev.trans (wrapper_returns _ _ _ _ _ _ _ _).2
  · intro cell
    dsimp only
    rw [mem,wrapper_word a _ _ sep]
    apply congrArg Sum.inr
    apply reference_current a _ _ _ sep
    intro p hp;exact alloc.frame p (Or.inl hp)
  · intro p hp outside
    dsimp only
    rw [mem,wrapper_full_frame _ _ _ _ _ _ _ _ _ outside]
    apply alloc.frame
    cases eq : r.out with
    | none=>left;intro h;rw [h] at hp;exact alloc.fresh eq hp
    | some t=>right;simp [eq]

#axiomsClean gemmNative_refines
#axiomsClean rowNative_refines
#axiomsClean gemm_native_complete
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
