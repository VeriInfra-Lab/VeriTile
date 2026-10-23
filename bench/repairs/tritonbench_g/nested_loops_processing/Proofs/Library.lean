import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Synchronization
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Shared.Storage

/-! Individual int32 tensor-library calls. All requests use signed Python
integers. Successful range/reshape replies describe the actual allocation;
errors keep the callback's actual after-state. No kernel result is an API law.
The inherited Stage tags below are only source-site labels (see BINDING.md). -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Host
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address (Dense)
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Execution (Disjoint)
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

structure Request where
  rows : Int
  cols : Int
  deriving DecidableEq

structure Flat (request : Request) where
  count : Nat
  count_eq : (count:Int)=request.rows*request.cols
  tensor : Packed [count]

/-- Because this reshape receives exactly arange(0, rows*cols), a successful
range followed by reshape cannot infer a negative requested dimension. A -1
with a positive partner would already have a negative arange endpoint; with
zero it is ambiguous; two negative dimensions are illegal. -/
structure Shaped (request : Request) where
  rows : Nat
  cols : Nat
  rows_eq : (rows:Int)=request.rows
  cols_eq : (cols:Int)=request.cols
  tensor : Packed [rows,cols]

theorem shaped_count (flat : Flat request) (x : Shaped request) : flat.count=x.rows*x.cols := by
  have h:=flat.count_eq
  rw [←x.rows_eq,←x.cols_eq] at h
  exact_mod_cast h

structure State (Store : Type) where
  memory : Nat → Store
  live : List Backing
  printed : Option (Σ shape,Packed shape)

def world (api : API Word Store Byte) (backings : List Backing) : World (State Store) Key Byte :=
  ⟨fun s=>observe api s.memory,caller backings⟩

structure Effect (api : API Word Store Byte) (backings : List Backing) (Value : Type) where
  call : Local (world api backings) Value
  retained : ∀ s b,b∈s.live → b∈(call.run s).1.live
  oldRead : ∀ (s : State Store) (b : Backing), b∈s.live → ∀ i, b.base ≤ i → i+4 ≤ b.base+b.nbytes →
    api.read ((call.run s).1.memory b.owner) i=api.read (s.memory b.owner) i

def Values (api : API Word Store Byte) (s : State Store) (p : Packed shape) (f : Nat → Word) : Prop :=
  p.backing∈s.live ∧ ∀ i,i<shape.prod → p.read api s.memory i=f i

theorem effect_packed (api : API Word Store Byte) (effect : Effect api backings Value)
    (s : State Store) (p : Packed shape) (live : p.backing∈s.live) (i : Nat) (bound : i<shape.prod) :
    p.read api (effect.call.run s).1.memory i=p.read api s.memory i := by
  apply effect.oldRead s p.backing live
  · simp [Packed.backing,Dense.address]
  · simp only [Packed.backing,Dense.address]
    omega

theorem effect_values (api : API Word Store Byte) (effect : Effect api backings Value)
    (s : State Store) (p : Packed shape) (f : Nat → Word) (values : Values api s p f) :
    Values api (effect.call.run s).1 p f :=
  ⟨effect.retained s _ values.1,fun i hi=>(effect_packed api effect s p values.1 i hi).trans (values.2 i hi)⟩

theorem call_ok {api : API Word Store Byte} (effect : Effect api backings Value)
    (stage : Stage) (s : State Store) (v : Value)
    (ok : ((call stage effect.call).run s).result=.ok v) : (effect.call.run s).2=.ok v := by
  cases h:(effect.call.run s).2 with
  | error e=>simp [call,h,Except.mapError] at ok
  | ok x=>simpa [call,h,Except.mapError] using ok

/-- Freshness and same-device layout are local allocator facts, not output
values. They refer to the actual zeros reply and include every caller byte. -/
structure Output (backings : List Backing) (x : Shaped request) where
  tensor : Packed [x.rows,x.cols]
  owner : tensor.owner=x.tensor.owner
  apart : Disjoint tensor.dense x.tensor.dense
  callerApart : ∀ key,caller backings key → ¬tensor.backing.contains key

def Output.placement (y : Output backings x) : Placement (x.rows*x.cols) :=
  ⟨⟨x.tensor.dense.base,by simpa using x.tensor.dense.finite,x.tensor.dense.aligned⟩,
   ⟨y.tensor.dense.base,by simpa using y.tensor.dense.finite,y.tensor.dense.aligned⟩,
   by simpa [VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Execution.Disjoint] using y.apart⟩

structure Number (value : Nat) where
  actual : Nat
  exact : actual=value

inductive DType where
  | int32
  deriving DecidableEq

structure Trial (count cols : Nat) where
  schedule : Schedule count cols
  completed : Nat
  failure : Option Nat
  failedWrites : List (Copy count)
  complete : failure=none → schedule.order.length≤completed
  synchronization : CtaBarrierContract

/-- On failure the runtime records the actual completed bounded word writes,
including any subset/order of lanes in a faulting vector store. The failure
frame theorem deliberately covers this overapproximation. -/
def Trial.writes (trial : Trial count cols) : List (Copy count) :=
  if trial.failure=none then trial.schedule.order.take trial.completed else trial.failedWrites

structure Library (api : API Word Store Byte) (backings : List Backing) where
  arange : (request : Request) → Effect api backings (Flat request)
  reshape : (request : Request) → Flat request → Effect api backings (Shaped request)
  device : {request : Request} → (x : Shaped request) → Effect api backings (Number x.tensor.owner)
  dtype : {request : Request} → (x : Shaped request) → Effect api backings DType
  zeros : {request : Request} → (x : Shaped request) → Number x.tensor.owner → DType → Effect api backings (Output backings x)
  numel : {request : Request} → (x : Shaped request) → Effect api backings (Number (x.rows*x.cols))
  stride0 : {request : Request} → (x : Shaped request) → Effect api backings (Number (max x.cols 1))
  stride1 : {request : Request} → (x : Shaped request) → Effect api backings (Number 1)
  setup : {request : Request} → (x : Shaped request) → Output backings x → Number (x.rows*x.cols) →
    Number (max x.cols 1) → Number 1 → Effect api backings (Trial (x.rows*x.cols) x.cols)
  finish : {request : Request} → (x : Shaped request) → Output backings x → Trial (x.rows*x.cols) x.cols → Effect api backings Unit
  print : {request : Request} → (x : Shaped request) → Output backings x → Effect api backings Unit
  arange_values : ∀ request s flat,((arange request).call.run s).2=.ok flat →
    Values api ((arange request).call.run s).1 flat.tensor arangeWord
  reshape_live : ∀ request flat s x,((reshape request flat).call.run s).2=.ok x →
    x.tensor.backing∈((reshape request flat).call.run s).1.live
  reshape_values : ∀ request flat s x,((reshape request flat).call.run s).2=.ok x →
    ∀ i,i<x.rows*x.cols → x.tensor.read api ((reshape request flat).call.run s).1.memory i=
      flat.tensor.read api s.memory i
  reshape_storage : ∀ request flat s x,((reshape request flat).call.run s).2=.ok x →
    x.tensor.owner=flat.tensor.owner ∧ x.tensor.dense.base=flat.tensor.dense.base
  zeros_values : ∀ request (x : Shaped request) device dtype s y,
    ((zeros x device dtype).call.run s).2=.ok y → Values api ((zeros x device dtype).call.run s).1 y.tensor (fun _=>0)
  zeros_fresh : ∀ request (x : Shaped request) device dtype s y,
    ((zeros x device dtype).call.run s).2=.ok y → ∀ key,y.tensor.backing.contains key →
      ∀ old∈s.live,¬old.contains key
  print_exact : ∀ request (x : Shaped request) y s,
    ((print x y).call.run s).2=.ok () → ((print x y).call.run s).1.printed=some ⟨[x.rows,x.cols],y.tensor⟩

/-- range and reshape are separate fallible calls in source order. -/
def input (api : API Word Store Byte) (lib : Library api backings) (request : Request) :
    Action (world api backings) (Shaped request) :=
  (call (.allocate .x) (lib.arange request).call).bind fun flat=>
    call (.shape .x) (lib.reshape request flat).call

theorem input_success (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (s : State Store) (x : Shaped request)
    (ok : ((input api lib request).run s).result=.ok x) :
    Values api ((input api lib request).run s).state x.tensor arangeWord := by
  unfold input at ok ⊢
  obtain ⟨flat,hf,hx,hs,_⟩:=bind_success _ _ s x ok
  have hf':=call_ok (lib.arange request) _ _ _ hf
  have hx':=call_ok (lib.reshape request flat) _ _ _ hx
  have values:=lib.arange_values request s flat hf'
  rw [hs]
  refine ⟨lib.reshape_live _ _ _ _ hx',?_⟩
  intro i hi
  exact (lib.reshape_values _ _ _ _ hx' i (by simpa using hi)).trans
    (values.2 i (by simpa [shaped_count flat x] using hi))

#axiomsClean shaped_count
#axiomsClean effect_values
#axiomsClean input_success
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Host
