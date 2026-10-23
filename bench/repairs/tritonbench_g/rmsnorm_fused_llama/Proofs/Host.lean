import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.FPGrid
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000

/-- The checked canonical FP32 domain of the repaired wrapper. The native
index bound includes formed addresses of inactive padded lanes. -/
structure Request where
  leading : List Nat
  columns : Nat
  epsilon : ℝ
  positive : 0<columns
  limit : columns≤16384
  positiveEpsilon : 0<epsilon
  indexFits : leading.prod*columns+16383<2^31
def Request.rows (q : Request) := q.leading.prod
def Request.shape (q : Request) := q.leading++[q.columns]
def Request.size (q : Request) := q.rows*q.columns

/-- A tensor descriptor whose region starts at its actual data pointer,
including any storage offset. Contiguous logical cells use linear offsets. -/
structure Tensor where
  region : RegionName
  shape : List Nat
  dtype : String
  elementBytes : Nat
  device : Nat
  contiguous : Bool

/-- Same-call ordinary tensor validation and fresh allocation. Allocation may
initialize arbitrary output cells; there is no kernel, output-value, reduction
or successful-execution premise. The frame also preserves input snapshots. -/
structure Preparation (σ C : Type) (memory : σ → RegionName → Nat → C) (q : Request) where
  before : σ
  prepared : σ
  input : Tensor
  weight : Tensor
  output : Tensor
  inputShape : input.shape=q.shape
  weightShape : weight.shape=[q.columns]
  outputShape : output.shape=q.shape
  types : input.dtype="float32" ∧ weight.dtype="float32" ∧ output.dtype="float32"
  widths : input.elementBytes=4 ∧ weight.elementBytes=4 ∧ output.elementBytes=4
  devices : weight.device=input.device ∧ output.device=input.device
  contiguous : input.contiguous=Bool.true ∧ weight.contiguous=Bool.true ∧ output.contiguous=Bool.true
  freshInput : input.region≠output.region
  freshWeight : weight.region≠output.region
  allocationFrame : ∀ r o, r≠output.region ∨ q.size≤o → memory prepared r o=memory before r o

structure Result (σ : Type) where
  tensor : Tensor
  state : σ

def invoke {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation (State α) (Cell α) State.mem q) : Option (Result (State α)) :=
  (Grid.run A call.input.region call.output.region call.weight.region q.rows q.columns 16384
    q.epsilon call.prepared q.rows).map (fun s => ⟨call.output,s⟩)

theorem whole_typed {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation (State α) (Cell α) State.mem q) :
    ∃ result, invoke A q call=some result ∧ result.tensor=call.output ∧
      result.tensor.shape=q.shape ∧ result.tensor.dtype="float32" ∧
      result.tensor.device=call.input.device ∧
      (∀ row, row<q.rows → ∀ k, k<q.columns →
        result.state.mem result.tensor.region (row*q.columns+k)=
          .mk .real (Row.value A call.before call.input.region call.weight.region row q.columns 16384 q.epsilon k)) ∧
      (∀ r o, r≠result.tensor.region ∨ q.size≤o → result.state.mem r o=call.before.mem r o) := by
  obtain ⟨s,he,hm⟩ := Grid.complete A call.input.region call.output.region call.weight.region
    q.rows q.columns 16384 q.epsilon call.prepared q.positive q.limit q.rows
  obtain ⟨hv,hf⟩ := Grid.graph_result A call.input.region call.output.region call.weight.region
    q.columns 16384 q.epsilon call.prepared q.limit call.freshInput call.freshWeight q.rows
  have hx : call.prepared.mem call.input.region=call.before.mem call.input.region :=
    funext (fun o => call.allocationFrame _ o (Or.inl call.freshInput))
  have hw : call.prepared.mem call.weight.region=call.before.mem call.weight.region :=
    funext (fun o => call.allocationFrame _ o (Or.inl call.freshWeight))
  refine ⟨⟨call.output,s⟩,by simp [invoke,he],rfl,call.outputShape,call.types.2.2,call.devices.2,?_,?_⟩
  · intro row hr k hk
    rw [hm,hv row hr k hk,Grid.value_congr A call.prepared call.before call.input.region
      call.weight.region row q.columns 16384 q.epsilon k hk hx hw]
  · intro r o h
    exact (congrFun (congrFun hm r) o).trans ((hf r o h).trans (call.allocationFrame r o h))

theorem empty_return {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation (State α) (Cell α) State.mem q) (h : q.rows=0) :
    invoke A q call=some ⟨call.output,call.prepared⟩ := by simp [invoke,h,Grid.run]

def invokeReal (q : Request) (call : Preparation BlockState MemCell BlockState.mem q) : Option (Result BlockState) :=
  (RealGrid.run call.input.region call.output.region call.weight.region q.rows q.columns 16384
    q.epsilon call.prepared q.rows).map (fun s => ⟨call.output,s⟩)

theorem whole_real (q : Request) (call : Preparation BlockState MemCell BlockState.mem q) :
    ∃ result, invokeReal q call=some result ∧ result.tensor=call.output ∧
      result.tensor.shape=q.shape ∧ result.tensor.dtype="float32" ∧
      result.tensor.device=call.input.device ∧
      (∀ row, row<q.rows → ∀ k, k<q.columns →
        result.state.readMem result.tensor.region (row*q.columns+k)=
          RealGrid.value (call.before.readMem call.input.region) (call.before.readMem call.weight.region)
            row q.columns q.epsilon k) ∧
      (∀ r o, r≠result.tensor.region ∨ q.size≤o → result.state.mem r o=call.before.mem r o) := by
  obtain ⟨s,he,hv,hf⟩ := RealGrid.complete call.input.region call.output.region call.weight.region
    q.rows q.columns 16384 q.epsilon call.prepared (by decide) call.freshInput call.freshWeight q.rows
  have hx : call.prepared.mem call.input.region=call.before.mem call.input.region :=
    funext (fun o => call.allocationFrame _ o (Or.inl call.freshInput))
  have hw : call.prepared.mem call.weight.region=call.before.mem call.weight.region :=
    funext (fun o => call.allocationFrame _ o (Or.inl call.freshWeight))
  refine ⟨⟨call.output,s⟩,by simp [invokeReal,he],rfl,call.outputShape,call.types.2.2,call.devices.2,?_,?_⟩
  · intro row hr k hk
    simpa [RealGrid.value,BlockState.readMem,hx,hw] using hv row hr k hk
  · intro r o h
    exact (hf r o h).trans (call.allocationFrame r o h)

#axiomsClean whole_typed
#axiomsClean whole_real
#axiomsClean empty_return
#specNonCircular Preparation avoiding [invoke,invokeReal,Grid.run,FP.Structural.exec,Triton.exec]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Host
