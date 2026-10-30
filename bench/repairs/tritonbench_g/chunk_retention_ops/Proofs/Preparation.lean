import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Host
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

/-- A logical read of an original Tensor uses its observed strides, including
zero strides and noncontiguous cotangents. The data pointer includes offset. -/
def tensorRead (c : Shape) (m : Memory α) (t : Contract.Tensor) (a : Address) : α :=
  m (t.metadata.dataPtr.toNat/4+
    (([a.head/c.H,a.head%c.H,a.row,a.col].zip t.strides).map
      fun p => p.1*p.2.toNat).sum)

def originalHeap (c : Shape) (zero : α) (m : Memory α)
    (objects : Region → Option Contract.Tensor) : Heap α := by
  classical
  exact fun a => if a.Valid c then
    match objects a.region with
    | some t => tensorRead c m t a
    | none => zero
    else zero

def forwardObjects (a : Contract.Forward) : Region → Option Contract.Tensor
  | .q => some a.q
  | .k => some a.k
  | .v => some a.v
  | .seed => a.initial
  | _ => none

def backwardObjects (a : Contract.Backward) : Region → Option Contract.Tensor
  | .incoming => a.incoming
  | .terminal => a.terminal
  | r => forwardObjects a.forward r

/-- Only per-element load/copy/zero laws for ordinary host preparation. No
history, dot, recurrence, gradient, or returned-result equation is admitted.
For forward these are allocator preservation and canonical-stride addressing;
for backward they additionally describe contiguous() and zeros_like(). -/
structure PreparedInputs (c : Shape) (storage : Storage c) (zero : α)
    (original prepared : Memory α) (objects : Region → Option Contract.Tensor) : Prop where
  load : ∀ a,a.Valid c → a.region∈inputRegions →
    (if storage.zeroInput a.region then zero else prepared (storage.binding.address a))=
    (match objects a.region with
     | some t => tensorRead c original t a
     | none => zero)

theorem PreparedInputs.read (p : PreparedInputs c storage zero original prepared objects)
    (a : Address) (input : a.region∈inputRegions) :
    storage.binding.view zero prepared a=originalHeap c zero original objects a := by
  classical
  by_cases valid : a.Valid c
  · have same := p.load a valid input
    cases hz : storage.zeroInput a.region <;>
      simpa [Binding.view,Storage.binding,originalHeap,valid,hz] using same
  · simp [Binding.view,originalHeap,valid]

theorem PreparedInputs.inputs (p : PreparedInputs c storage zero original prepared objects) :
    inputsOfHeap (storage.binding.view zero prepared)=inputsOfHeap (originalHeap c zero original objects) := by
  unfold inputsOfHeap
  congr 1 <;> funext h r j <;> apply p.read <;> simp [inputRegions]

def backwardCallers (a : Contract.Backward) :=
  callerInputs a.forward++a.incoming.toList++a.terminal.toList

/-- Actual saved handles, normalized cotangent allocations, allocator
freshness, and the exact Python branch predicates. -/
structure BackwardCall (a : Contract.Backward) (admitted : Contract.BackwardContract a)
    (needsInitial : Bool) extends
    BackwardAllocation (shape a.forward admitted.1)
      (a.forward.initial.isSome && needsInitial) (decide (a.forward.checkpoint=1)) where
  device : storage.device=a.forward.k.metadata.device
  q : storage.tensor .q=a.forward.q
  k : storage.tensor .k=a.forward.k
  v : storage.tensor .v=a.forward.v
  primals : ∀ r∈[Region.q,Region.k,Region.v],storage.present r=Bool.true
  seed : storage.zeroInput .seed= !a.forward.initial.isSome
  initialInput : ∀ t∈a.forward.initial.toList,storage.tensor .seed=t ∧ storage.present .seed=Bool.true
  incoming : storage.present .incoming=Bool.true
  incomingRead : storage.zeroInput .incoming=Bool.false
  terminal : a.terminal.isSome=Bool.true → storage.present .terminal=Bool.true
  terminalRead : storage.zeroInput .terminal= !a.terminal.isSome
  freshOriginal : ∀ t∈backwardCallers a,FreshCaller storage t

theorem BackwardCall.initial_metadata (call : BackwardCall a admitted needsInitial)
    (enabled : (a.forward.initial.isSome && needsInitial)=Bool.true) :
    (backwardTensors call.storage (a.forward.initial.isSome && needsInitial))[3]?=
      some (some (call.storage.tensor .initialGradient)) ∧
    (call.storage.tensor .initialGradient).metadata.shape=Contract.stateShape a.forward.dims ∧
    (call.storage.tensor .initialGradient).metadata.dtype=.floating .fp32 ∧
    (call.storage.tensor .initialGradient).metadata.device=a.forward.k.metadata.device ∧
    (call.storage.tensor .initialGradient).metadata.isContiguous=Bool.true := by
  obtain ⟨hs,hd,hdev,hc⟩ := returned_metadata call.storage .initialGradient (call.initial enabled)
  exact ⟨by simp [backwardTensors,enabled],hs,hd,hdev.trans call.device,hc⟩

theorem backward_all_stores_valid (c : Shape) (tree : SumTree) (ri checkpoint : Bool)
    (s : Store) (mem : s∈recomputeCode c tree checkpoint++backwardCode c tree ri) :
    s.address.Valid c ∧ s.address.region∉inputRegions := by
  rcases List.mem_append.mp mem with h|h
  · exact recompute_stores_valid c tree checkpoint s h
  · exact backward_stores_valid c tree ri s h

theorem backward_backing_frame (abi : MemoryABI σ) (ports : Numeric.ScalarPorts)
    (a : Contract.Backward) (admitted : Contract.BackwardContract a)
    (call : BackwardCall a admitted needsInitial) (scale : Word)
    (compiler : Compiler (shape a.forward admitted.1) tree a.forward.scale.value
      a.forward.initial.isSome a.forward.final a.terminal.isSome (a.forward.initial.isSome && needsInitial))
    (original prepared : σ)
    (allocationFrame : ∀ t∈backwardCallers a,∀ byte,byte<t.bytes →
      abi.bytes prepared (t.base+byte)=abi.bytes original (t.base+byte)) :
    ∀ t∈backwardCallers a,∀ byte,byte<t.bytes →
      abi.bytes (backwardEntry abi ports _ call.storage scale compiler
        (decide (a.forward.checkpoint=1)) prepared).2 (t.base+byte)=
      abi.bytes original (t.base+byte) := by
  intro t member byte bound
  change abi.bytes (nativeExecute abi ports _ call.storage.binding scale _ prepared) _=_
  rw [compiler.backward_eq,
    complete_backing_frame abi ports _ call.storage scale _
      (backward_all_stores_valid _ tree _ _) (call.toBackwardAllocation.allocated tree)
      prepared t (call.freshOriginal t member) byte bound,
    allocationFrame t member byte bound]

#axiomsClean PreparedInputs.inputs
#axiomsClean backward_backing_frame
#specNonCircular PreparedInputs avoiding [Reference.output,Reference.dq,Reference.dk,Reference.dv]
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
