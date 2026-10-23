import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectPointers
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
noncomputable section
set_option maxHeartbeats 1500000

/-- Frontend/compiler correspondence contains only source identity, the
selected local instruction plan and complete dispatch. Native arithmetic and
grid bounds are supplied by the checked contract, not assumed here. -/
structure SourceCall (I F H : Type) (r : Request) (out : Nat) (divisor : F) where
  invocation : Invocation (requestShape r)
  admitted : Source.Admitted invocation.config
  parsedProduction : String
  productionIdentity : parsedProduction=Source.productionAST
  jit : VeriTile.Triton.ComputeKernel
  jitIdentity : jit=Source.typedJIT (requestShape r) invocation.config
    (canonical (requestShape r).M (requestShape r).K) (canonical (requestShape r).K (requestShape r).N)
    (canonical (requestShape r).M (requestShape r).N) (hasBias r)
  dispatch : invocation.schedule.events.Perm (allCommits (requestShape r) invocation.config)
  lowered : Code I F H Unit
  sourceLowering : lowered=gridCode (requestShape r) invocation.config (requestBuffers r out)
    invocation.selected divisor (hasBias r) invocation.schedule.events

def SourceCall.bound (call : SourceCall I F H r out divisor)
    (active : ¬((requestShape r).M=0 ∨ (requestShape r).N=0)) :
    BoundCall I F H (requestShape r) (requestBuffers r out) divisor (hasBias r) where
  invocation := call.invocation
  aDense := canonical (requestShape r).M (requestShape r).K
  bDense := canonical (requestShape r).K (requestShape r).N
  cDense := canonical (requestShape r).M (requestShape r).N
  outputStrides := ⟨rfl,rfl⟩
  compiler := {
    parsedProduction:=call.parsedProduction, productionIdentity:=call.productionIdentity,
    jit:=call.jit, jitIdentity:=call.jitIdentity, admitted:=call.admitted,
    addressABI:=(native_admitted r.a.info r.b.info r.x.info r.w.info _ out r.contract
      (by change ¬(rows r.a.info=0 ∨ cols r.b.info=0) at active;omega)
      (by change ¬(rows r.a.info=0 ∨ cols r.b.info=0) at active;omega) call.invocation.config call.admitted).1,
    gridABI:=(native_admitted r.a.info r.b.info r.x.info r.w.info _ out r.contract
      (by change ¬(rows r.a.info=0 ∨ cols r.b.info=0) at active;omega)
      (by change ¬(rows r.a.info=0 ∨ cols r.b.info=0) at active;omega) call.invocation.config call.admitted).2,
    dispatch:=call.dispatch,lowered:=call.lowered,sourceLowering:=call.sourceLowering }

/-- Empty output bypasses compilation and launches. No unused IndexABI or
GridABI witness is required for dimensions in that branch. -/
def native (r : Request) (out : Nat) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (divisor : F) (callerDevice : Nat) (benchmarks : List (SourceCall I F H r out divisor))
    (chosen : SourceCall I F H r out divisor) (state : σ) : NativeReturned σ :=
  if empty : (requestShape r).M=0 ∨ (requestShape r).N=0 then
    ⟨outputTensor (requestShape r) (requestBuffers r out) r.a.info.metadata.device.index,state,callerDevice⟩
  else compiledWrapper abi hook r.a.info.metadata.device.index callerDevice
    (benchmarks.map (fun c=>c.bound empty)) (chosen.bound empty) state

theorem native_refines (r : Request) (out : Nat) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (divisor : F) (callerDevice : Nat) (benchmarks : List (SourceCall I F H r out divisor))
    (chosen : SourceCall I F H r out divisor) (state : σ) :
    let actual:=native r out abi hook divisor callerDevice benchmarks chosen state
    let reference:=wrapper a (requestShape r) (requestBuffers r out) divisor (hasBias r)
      r.a.info.metadata.device.index callerDevice (benchmarks.map SourceCall.invocation) chosen.invocation (abi.observe state)
    (actual.tensor,abi.observe actual.state,actual.currentDevice)=
      (reference.tensor,reference.memory,reference.currentDevice) := by
  unfold native
  split_ifs with empty
  · simp only [wrapper,if_pos empty]
  · simpa only [List.map_map] using compiledWrapper_refines abi hook r.a.info.metadata.device.index callerDevice
      (benchmarks.map (fun c=>c.bound empty)) (chosen.bound empty) state

structure NativeAllocation (r : Request) (abi : LocalEventABI I F H σ a) (caller : σ) (live : Nat → Prop) where
  storage : Nat
  state : σ
  fresh : ¬live storage
  frame : ∀ p,p.1≠storage → abi.observe state p=abi.observe caller p

def NativeAllocation.logical (alloc : NativeAllocation r abi caller live) : Allocation r (abi.observe caller) live :=
  ⟨alloc.storage,abi.observe alloc.state,alloc.fresh,alloc.frame⟩

def checkedNative (r : Request) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (divisor : F) (callerDevice : Nat) (caller : σ) (live : Nat → Prop) (alloc : NativeAllocation r abi caller live)
    (benchmarks : List (SourceCall I F H r alloc.storage divisor)) (chosen : SourceCall I F H r alloc.storage divisor) :
    Except InputError (NativeReturned σ) := do
  checkTensors r.a.info r.b.info r.x.info r.w.info (r.bias.map TensorView.info)
  return native r alloc.storage abi hook divisor callerDevice benchmarks chosen alloc.state

theorem native_complete (r : Request) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (divisor : F) (callerDevice : Nat) (caller : σ) (live : Nat → Prop) (inputs : InputsLive r live)
    (alloc : NativeAllocation r abi caller live)
    (benchmarks : List (SourceCall I F H r alloc.storage divisor)) (chosen : SourceCall I F H r alloc.storage divisor) :
    ∃ result, checkedNative r abi hook divisor callerDevice caller live alloc benchmarks chosen=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell (requestShape r),abi.observe result.state (outputAddress (requestShape r) (requestBuffers r alloc.storage) cell)=
        .inr (.inr (wordReference a (requestShape r) chosen.invocation.config (requestBuffers r alloc.storage) chosen.invocation.selected divisor
          (hasBias r) chosen.invocation.schedule (abi.observe caller) cell))) ∧
      (∀ p,live p.1 → abi.observe result.state p=abi.observe caller p) := by
  have refinement:=native_refines r alloc.storage abi hook divisor callerDevice benchmarks chosen alloc.state
  have tensor:=congrArg Prod.fst refinement
  have memory:=congrArg (fun t=>t.2.1) refinement
  have device:=congrArg (fun t=>t.2.2) refinement
  simp only [Prod.fst,Prod.snd] at tensor memory device
  refine ⟨native r alloc.storage abi hook divisor callerDevice benchmarks chosen alloc.state,?_,?_,?_,?_,?_⟩
  · simp [checkedNative,(checkTensors_iff _ _ _ _ _).mpr r.contract,Except.bind,Except.pure,bind,pure]
  · exact tensor.trans (wrapper_returns _ _ _ _ _ _ _ _ _ _).1
  · exact device.trans (wrapper_returns _ _ _ _ _ _ _ _ _ _).2
  · intro cell
    have fresh : (requestBuffers r alloc.storage).Fresh:=allocation_fresh r _ live inputs alloc.logical
    rw [memory,wrapper_word _ _ _ fresh]
    apply congrArg (fun value=>Sum.inr (Sum.inr value))
    exact allocated_reference r a divisor chosen.invocation (abi.observe caller) live inputs alloc.logical cell
  · intro p hp
    rw [memory]
    exact caller_frame a _ _ _ _ _ _ _ _ _ _ live (allocationABI r _ live alloc.logical) p hp

#axiomsClean native_refines
#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
