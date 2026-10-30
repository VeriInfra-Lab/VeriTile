import bench.tritonbench_g.chunk_delta_fwd.Proofs.Bounds
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealOutputs
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Host
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000

/-- Addresses and capacities are in FP32 elements. Tensor identity is retained
separately from its data pointer, as for ordinary views. -/
structure Tensor where
  (identity base capacity device : Nat)
  (shape strides : List Nat)
  dtype : ComputeDType
  ordinary : Bool
  contiguous : Bool
  deriving DecidableEq
def denseStrides : List Nat → List Nat
  | [] => []
  | _::rest => rest.prod::denseStrides rest
def span (t : Tensor) := ((t.shape.zip t.strides).map (fun z => (z.1-1)*z.2)).sum+1
def separate (a : Tensor) (aSize : Nat) (b : Tensor) (bSize : Nat) : Prop :=
  a.base+aSize ≤ b.base ∨ b.base+bSize ≤ a.base

structure Call where
  dims : Dimensions
  (key weight value : Tensor)
  (initial final : Option Tensor)
def Call.inputs (call : Call) : List Tensor := [call.key,call.weight,call.value]++call.initial.toList

structure CallContract (call : Call) : Prop where
  valid : Valid call.dims
  bounds : AddressContract call.dims
  seedFlag : call.dims.seed=call.initial.isSome
  finalFlag : call.dims.final=call.final.isSome
  keyShape : call.key.shape=[call.dims.B,call.dims.H,call.dims.T,call.dims.K]
  weightShape : call.weight.shape=call.key.shape
  valueShape : call.value.shape=correctedShape call.dims
  keyStride : call.key.strides=denseStrides call.key.shape
  valueStride : call.value.strides=denseStrides call.value.shape
  weightStride : call.weight.strides=[call.dims.wBatch,call.dims.wHead,call.dims.wTime,call.dims.wDim]
  states : ∀ t∈call.initial.toList++call.final.toList,
    t.shape=stateShape call.dims ∧ t.contiguous=Bool.true
  stateSpan : ∀ t∈call.initial.toList++call.final.toList, span t=t.shape.prod
  storage : ∀ t∈call.inputs++call.final.toList,
    t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧ t.device=call.key.device ∧ span t ≤ t.capacity
  finalSeparated : ∀ f∈call.final, ∀ t∈call.inputs, separate f f.shape.prod t (span t)

structure Allocations where
  (history corrected : Tensor)

/-- Ordinary successful allocator contract. Freshness concerns storage and
identity; it contains no computed value, loop, dot, or output premise. -/
structure AllocationContract (call : Call) (a : Allocations) : Prop where
  historyShape : a.history.shape=Host.historyShape call.dims
  correctedShape : a.corrected.shape=Host.correctedShape call.dims
  descriptors : ∀ t∈[a.history,a.corrected], t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧
    t.device=call.key.device ∧ t.strides=denseStrides t.shape ∧ t.shape.prod ≤ t.capacity
  fresh : ∀ t∈[a.history,a.corrected], ∀ u∈call.inputs++call.final.toList,
    separate t t.shape.prod u (span u) ∧ t.identity≠u.identity
  pairSeparated : separate a.history a.history.shape.prod a.corrected a.corrected.shape.prod
  pairIdentity : a.history.identity≠a.corrected.identity

def descriptor (call : Call) (a : Allocations) (r : RegionName) : Option Tensor :=
  if r="key" then some call.key else if r="weight" then some call.weight else
  if r="value" then some call.value else if r="initial" then call.initial else
  if r="final" then call.final else if r="history" then some a.history else
  if r="corrected" then some a.corrected else none
def logical {α : Type} (call : Call) (a : Allocations) (physical : Nat → Cell α)
    (fallback : Memory α) : Memory α := fun r o =>
  match descriptor call a r with | some t => physical (t.base+o) | none => fallback r o

structure Result (α : Type) where
  memory : Memory α
  returned : Tensor × Tensor

def execute {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (call : Call) (a : Allocations) (schedule : List PID) (m : Memory α) : Option (Result α) := do
  let after ← runGrid M D call.dims.parameters schedule m
  return ⟨after,(a.history,a.corrected)⟩

structure Post {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (call : Call) (cc : CallContract call) (a : Allocations) (before : Memory α)
    (result : Result α) : Prop where
  returned : result.returned=(a.history,a.corrected)
  historyDescriptor : result.returned.1.shape=historyShape call.dims
  correctedDescriptor : result.returned.2.shape=correctedShape call.dims
  values : GridSpec M D call.dims.parameters (shape_valid call.dims cc.valid) before before result.memory

theorem execute_complete {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (call : Call) (cc : CallContract call) (a : Allocations) (ac : AllocationContract call a)
    (schedule : List PID) (hs : schedule.Perm (grid call.dims.parameters)) (before : Memory α) :
    ∃ result, execute M D call a schedule before=some result ∧ Post M D call cc a before result := by
  obtain ⟨after,he,hg⟩ := whole_grid M D call.dims.parameters (shape_valid call.dims cc.valid)
    schedule hs before before (SameInputs.refl before)
  refine ⟨⟨after,(a.history,a.corrected)⟩,?_,rfl,ac.historyShape,ac.correctedShape,hg⟩
  simp [execute,he]

theorem real_complete (call : Call) (cc : CallContract call) (a : Allocations)
    (ac : AllocationContract call a) (schedule : List PID)
    (hs : schedule.Perm (grid call.dims.parameters)) (before : Memory ℝ) :
    ∃ result, execute Real.algebra Real.dot call a schedule before=some result ∧
      result.returned=(a.history,a.corrected) ∧
      Real.Spec call.dims.parameters before before result.memory := by
  obtain ⟨result,he,hp⟩ := execute_complete Real.algebra Real.dot call cc a ac schedule hs before
  exact ⟨result,he,hp.returned,Real.grid_to_real _ _ _ _ _ hp.values⟩

#axiomsClean execute_complete
#axiomsClean real_complete
#specNonCircular AllocationContract avoiding [execute,runGrid,states]
#specNonCircular Post avoiding [execute,runGrid,gridMemory]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Host
