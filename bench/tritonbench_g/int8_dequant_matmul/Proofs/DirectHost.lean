import bench.tritonbench_g.int8_dequant_matmul.Proofs.Int8DequantMatmul_Direct_Contract
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Bound
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
noncomputable section
set_option maxHeartbeats 1200000

/-- Original tensor object and storage-index map. No materialized copy. -/
structure TensorView where
  info : TensorInfo
  object : Nat
  physicalIndex : Nat → Nat

structure TensorLaws (bytes : Nat) (storageBase : Nat → Nat) (t : TensorView) : Prop where
  pointer : 0<t.info.metadata.numel →
    t.info.metadata.dataPtr=BitVec.ofNat 64 (storageBase t.info.storage+bytes*t.info.offset)
  contiguous : t.info.metadata.isContiguous=Bool.true → ∀ i,i<t.info.metadata.numel → t.physicalIndex i=t.info.offset+i
  capacity : 0<t.info.metadata.numel → storageBase t.info.storage+bytes*(t.info.offset+t.info.metadata.numel)≤2^64

structure Request where
  a : TensorView
  b : TensorView
  x : TensorView
  w : TensorView
  bias : Option TensorView
  contract : InputContract a.info b.info x.info w.info (bias.map TensorView.info)

def requestShape (r : Request) := shape r.a.info r.b.info
def requestBuffers (r : Request) (out : Nat) := buffers r.a.info r.b.info r.x.info r.w.info (r.bias.map TensorView.info) out
def hasBias (r : Request) : Bool := r.bias.isSome

structure InputsLive (r : Request) (live : Nat → Prop) : Prop where
  a : live r.a.info.storage
  b : live r.b.info.storage
  x : live r.x.info.storage
  w : live r.w.info.storage
  bias : ∀ t,r.bias=some t → live t.info.storage

/-- Ordinary fresh FP16 empty allocation. The pointer/size facts are allocator
semantics; it does not contain any kernel-result or numeric obligation. -/
structure Allocation (r : Request) (caller : Memory F H) (live : Nat → Prop) where
  storage : Nat
  memory : Memory F H
  fresh : ¬live storage
  frame : ∀ p,p.1≠storage → memory p=caller p

theorem allocation_fresh (r : Request) (caller : Memory F H) (live : Nat → Prop)
    (inputs : InputsLive r live) (alloc : Allocation r caller live) : (requestBuffers r alloc.storage).Fresh := by
  have sep : ∀ s,live s → s≠alloc.storage := by intro s hs eq;rw [eq] at hs;exact alloc.fresh hs
  refine ⟨sep _ inputs.a,sep _ inputs.b,sep _ inputs.x,sep _ inputs.w,?_⟩
  cases eq : r.bias with
  | none=>simpa [requestBuffers,buffers,eq] using sep _ inputs.x
  | some t=>simpa [requestBuffers,buffers,eq] using sep _ (inputs.bias t eq)

theorem allocationABI (r : Request) (caller : Memory F H) (live : Nat → Prop)
    (alloc : Allocation r caller live) : AllocationABI (requestBuffers r alloc.storage) caller alloc.memory live := by
  refine ⟨alloc.fresh,?_⟩
  intro p hp
  apply alloc.frame
  intro eq;rw [eq] at hp;exact alloc.fresh hp

def run (r : Request) (a : Ops I F H) (divisor : F) (callerDevice : Nat)
    (benchmarks : List (Invocation (requestShape r))) (chosen : Invocation (requestShape r))
    (caller : Memory F H) (live : Nat → Prop) (alloc : Allocation r caller live) : Except InputError (Returned F H) := do
  checkTensors r.a.info r.b.info r.x.info r.w.info (r.bias.map TensorView.info)
  return wrapper a (requestShape r) (requestBuffers r alloc.storage) divisor (hasBias r)
    r.a.info.metadata.device.index callerDevice benchmarks chosen alloc.memory

theorem run_ok (r : Request) (a : Ops I F H) (divisor : F) (callerDevice : Nat)
    (benchmarks : List (Invocation (requestShape r))) (chosen : Invocation (requestShape r))
    (caller : Memory F H) (live : Nat → Prop) (alloc : Allocation r caller live) :
    run r a divisor callerDevice benchmarks chosen caller live alloc=.ok
      (wrapper a (requestShape r) (requestBuffers r alloc.storage) divisor (hasBias r)
        r.a.info.metadata.device.index callerDevice benchmarks chosen alloc.memory) := by
  simp [run,(checkTensors_iff _ _ _ _ _).mpr r.contract,Except.bind,bind,pure,Except.pure]

theorem allocated_reference (r : Request) (a : Ops I F H) (divisor : F)
    (v : Invocation (requestShape r)) (caller : Memory F H) (live : Nat → Prop)
    (inputs : InputsLive r live) (alloc : Allocation r caller live) (cell : MatrixCell (requestShape r)) :
    wordReference a (requestShape r) v.config (requestBuffers r alloc.storage) v.selected divisor (hasBias r)
      v.schedule alloc.memory cell =
    wordReference a (requestShape r) v.config (requestBuffers r alloc.storage) v.selected divisor (hasBias r)
      v.schedule caller cell := by
  apply reference_current a _ _ _ (allocation_fresh r caller live inputs alloc)
  exact alloc.frame

theorem run_complete (r : Request) (a : Ops I F H) (divisor : F) (callerDevice : Nat)
    (benchmarks : List (Invocation (requestShape r))) (chosen : Invocation (requestShape r))
    (caller : Memory F H) (live : Nat → Prop) (inputs : InputsLive r live) (alloc : Allocation r caller live) :
    ∃ result, run r a divisor callerDevice benchmarks chosen caller live alloc=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell (requestShape r),result.memory (outputAddress (requestShape r) (requestBuffers r alloc.storage) cell)=
        .inr (.inr (wordReference a (requestShape r) chosen.config (requestBuffers r alloc.storage) chosen.selected divisor
          (hasBias r) chosen.schedule caller cell))) ∧
      (∀ p,live p.1 → result.memory p=caller p) := by
  refine ⟨_,run_ok r a divisor callerDevice benchmarks chosen caller live alloc,?_,?_,?_,?_⟩
  · exact (wrapper_returns _ _ _ _ _ _ _ _ _ _).1
  · exact (wrapper_returns _ _ _ _ _ _ _ _ _ _).2
  · intro cell
    rw [wrapper_word _ _ _ (allocation_fresh r caller live inputs alloc),
      allocated_reference r a divisor chosen caller live inputs alloc cell]
  · intro p hp
    exact caller_frame a _ _ _ _ _ _ _ _ _ _ live (allocationABI r caller live alloc) p hp

#axiomsClean run_complete

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
