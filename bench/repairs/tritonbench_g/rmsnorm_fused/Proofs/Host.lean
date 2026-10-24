import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.RmsnormFused.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

structure Request where
  geometry : Shape
  valid : geometry.Valid
  gridFits : geometry.rows≤2^31-1
  epsilon : ℝ
  positiveEpsilon : 0<epsilon
def Request.rows (q : Request) := q.geometry.rows
def Request.columns (q : Request) := q.geometry.width
def Request.block (q : Request) := q.geometry.block
def Request.shape (q : Request) := q.geometry.dims
def Request.size (q : Request) := q.rows*q.columns

structure Tensor where
  region : RegionName
  dataPtr : Nat
  shape : List Nat
  dtype : String
  elementBytes : Nat
  device : Nat
  cuda : Bool
  layout : String
  contiguous : Bool
  storageOffset : Nat
  lazyNeg : Bool
  lazyConj : Bool

/-- Same-call tensor validation and ordinary fresh allocation. Neither values
computed by the kernel nor successful execution occur in this boundary. -/
structure Preparation (σ C : Type) (memory : σ → RegionName → Nat → C) (q : Request) where
  before : σ
  prepared : σ
  input : Tensor
  weight : Tensor
  output : Tensor
  callerDevice : Nat
  inputShape : input.shape=q.shape
  weightShape : weight.shape=[q.columns]
  outputShape : output.shape=q.shape
  types : input.dtype="float32" ∧ weight.dtype="float32" ∧ output.dtype="float32"
  widths : input.elementBytes=4 ∧ weight.elementBytes=4 ∧ output.elementBytes=4
  devices : weight.device=input.device ∧ output.device=input.device
  cuda : input.cuda=Bool.true ∧ weight.cuda=Bool.true ∧ output.cuda=Bool.true
  layouts : input.layout="strided" ∧ weight.layout="strided" ∧ output.layout="strided"
  contiguous : input.contiguous=Bool.true ∧ weight.contiguous=Bool.true ∧ output.contiguous=Bool.true
  offsets : input.storageOffset=0 ∧ weight.storageOffset=0 ∧ output.storageOffset=0
  flags : input.lazyNeg=Bool.false ∧ input.lazyConj=Bool.false ∧
    weight.lazyNeg=Bool.false ∧ weight.lazyConj=Bool.false ∧
    output.lazyNeg=Bool.false ∧ output.lazyConj=Bool.false
  freshInput : input.region≠output.region
  freshWeight : weight.region≠output.region
  allocationFrame : ∀ r o,r≠output.region ∨ q.size≤o → memory prepared r o=memory before r o
  /-- Ordinary live-storage spans from the actual tensor data pointers. -/
  inputSpan : q.rows≠0 → input.dataPtr+4*q.size≤2^64
  weightSpan : q.rows≠0 → weight.dataPtr+4*q.columns≤2^64
  outputSpan : q.rows≠0 → output.dataPtr+4*q.size≤2^64
  outputAlignment : q.rows≠0 → output.dataPtr%4=0

structure Result (σ : Type) where
  tensor : Tensor
  state : σ
  currentDevice : Nat
  launchDevice : Option Nat

/-- Owner device selection and restoration on normal completion of the Python
context manager. Empty leading dimensions perform no launch. -/
def finish {σ C : Type} {memory : σ → RegionName → Nat → C} {q : Request}
    (call : Preparation σ C memory q) (s : σ) : Result σ :=
  ⟨call.output,s,call.callerDevice,if q.rows=0 then none else some call.input.device⟩

def invoke {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation (State α) (Cell α) State.mem q) (schedule : List Nat) :
    Option (Result (State α)) :=
  (Row.source A call.input.region call.output.region call.weight.region q.rows q.columns q.block
    q.epsilon schedule call.prepared).map (finish call)

theorem whole_typed {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation (State α) (Cell α) State.mem q) (schedule : List Nat)
    (all : schedule.Perm q.geometry.grid) :
    ∃ result,invoke A q call schedule=some result ∧ result.tensor=call.output ∧
      result.tensor.shape=q.shape ∧ result.tensor.dtype="float32" ∧
      result.tensor.device=call.input.device ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.rows=0 then none else some call.input.device) ∧
      (∀ row k,row<q.rows → k<q.columns →
        (result.state.mem result.tensor.region (row*q.columns+k)).read .real=
          Row.value A call.before call.input.region call.weight.region row q.columns q.block q.epsilon k) ∧
      (∀ r o,r≠result.tensor.region ∨ q.size≤o → result.state.mem r o=call.before.mem r o) := by
  obtain ⟨s,he,hm⟩ := Row.source_complete A call.input.region call.output.region call.weight.region
    q.rows q.columns q.block q.epsilon q.valid.1 q.valid.2.1 schedule call.prepared
  have inputs : Row.InputsEq call.input.region call.weight.region call.prepared call.before :=
    ⟨fun o => call.allocationFrame _ o (Or.inl call.freshInput),
     fun o => call.allocationFrame _ o (Or.inl call.freshWeight)⟩
  refine ⟨finish call s,by simp [invoke,he],rfl,call.outputShape,call.types.2.2,
    call.devices.2,rfl,rfl,?_,?_⟩
  · intro row k hr hk
    change (s.mem call.output.region (row*q.columns+k)).read .real=_
    rw [hm]
    exact (Row.whole_value A q.geometry q.valid _ _ _ q.epsilon schedule all call.prepared
      row k hr hk call.freshInput call.freshWeight).trans
      (Row.value_congr A call.prepared call.before _ _ row q.columns q.block k q.epsilon inputs hk)
  · intro r o h
    change s.mem r o=call.before.mem r o
    rw [hm]
    apply (Row.grid_frame A _ _ _ q.columns q.block q.epsilon schedule call.prepared r o ?_).trans
      (call.allocationFrame r o h)
    rcases h with h|h
    · exact Or.inl h
    · right
      intro row hr k hk eq
      have bound := offset_bounded q.geometry row k
        ((grid_complete q.geometry row).mp (all.mem_iff.mp hr)) hk
      change row*q.columns+k<q.size at bound
      omega

theorem empty_return {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation (State α) (Cell α) State.mem q) (schedule : List Nat)
    (all : schedule.Perm q.geometry.grid) (h : q.rows=0) :
    invoke A q call schedule=some ⟨call.output,call.prepared,call.callerDevice,none⟩ := by
  have hs : schedule=[] := by
    have hgeom : q.geometry.rows=0 := h
    have hl : schedule.length=0 := by
      simpa only [Shape.grid,List.length_range,hgeom] using all.length_eq
    exact List.eq_nil_of_length_eq_zero hl
  simp [invoke,hs,Row.source,finish,h]

def invokeReal (q : Request) (call : Preparation BlockState MemCell BlockState.mem q)
    (schedule : List Nat) : Option (Result BlockState) :=
  (Real.source call.input.region call.output.region call.weight.region q.rows q.columns q.block
    q.epsilon schedule call.prepared).map (finish call)

theorem whole_real (q : Request) (call : Preparation BlockState MemCell BlockState.mem q)
    (schedule : List Nat) (all : schedule.Perm q.geometry.grid) :
    ∃ result,invokeReal q call schedule=some result ∧ result.tensor=call.output ∧
      result.tensor.shape=q.shape ∧ result.tensor.dtype="float32" ∧
      result.tensor.device=call.input.device ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.rows=0 then none else some call.input.device) ∧
      (∀ row k,row<q.rows → k<q.columns →
        result.state.readMem result.tensor.region (row*q.columns+k)=
          Real.formula call.before call.input.region call.weight.region row q.columns q.epsilon k) ∧
      (∀ r o,r≠result.tensor.region ∨ q.size≤o → result.state.mem r o=call.before.mem r o) := by
  obtain ⟨s,he,hv,hf⟩ := Real.whole q.geometry q.valid call.input.region call.output.region
    call.weight.region q.epsilon call.freshInput call.freshWeight schedule all call.prepared
  refine ⟨finish call s,by simp [invokeReal,Request.rows,Request.columns,Request.block,he],rfl,call.outputShape,call.types.2.2,
    call.devices.2,rfl,rfl,?_,?_⟩
  · intro row k hr hk
    exact (hv row k hr hk).trans (Real.formula_congr call.prepared call.before _ _ row q.columns q.epsilon k
      (fun o => call.allocationFrame _ o (Or.inl call.freshInput))
      (fun o => call.allocationFrame _ o (Or.inl call.freshWeight)))
  · intro r o h
    exact (hf r o h).trans (call.allocationFrame r o h)

#axiomsClean whole_typed
#axiomsClean whole_real
#axiomsClean empty_return
#specNonCircular Preparation avoiding [invoke,invokeReal,Row.source,FP.Structural.exec,Triton.exec]
end
end VeriTile.Bench.Repairs.RmsnormFused.Host
