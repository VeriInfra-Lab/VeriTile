import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Machine
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Events

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

inductive DType where
  | int8 | fp32 | fp16
  deriving DecidableEq
structure Tensor where
  storage : Nat
  offset : Nat
  shape : List Nat
  strides : List Nat
  dtype : DType
  device : Nat
  deriving DecidableEq
def outputTensor (s : Shape) (b : Buffers) (device : Nat) : Tensor :=
  ⟨b.out,b.outOffset,[s.M,s.N],[max s.N 1,1],.fp16,device⟩

/-- Ordinary caller admission. FP32 is required for unspecified float inputs;
signed int8 and FP16 output remain the algorithm's required storage formats. -/
structure Admission (s : Shape) (A B X W : Tensor) (bias : Option Tensor) (owner : Nat) : Prop where
  aShape : A.shape=[s.M,s.K]
  bShape : B.shape=[s.K,s.N]
  xShape : X.shape=[s.M]
  wShape : W.shape=[s.N]
  aType : A.dtype=.int8
  bType : B.dtype=.int8
  xType : X.dtype=.fp32
  wType : W.dtype=.fp32
  biasType : ∀ t,bias=some t → t.dtype=.fp32 ∧ t.shape=[s.N] ∧ t.device=owner
  devices : A.device=owner ∧ B.device=owner ∧ X.device=owner ∧ W.device=owner

/-- The actual wrapper's conditional .contiguous() may also run for a
singleton whose two inert strides both exceed one. Its ordinary Tensor copy
contract is per-byte transport into a fresh allocation. This contract is
separate from all dot/FP reasoning. -/
structure ResolvedMatrix (rows cols : Nat) (original resolved : Tensor)
    (caller prepared : Memory F H) where
  originalDense : Dense2 rows cols
  resolvedDense : Dense2 rows cols
  originalStrides : original.strides=[originalDense.rowStride,originalDense.colStride]
  resolvedStrides : resolved.strides=[resolvedDense.rowStride,resolvedDense.colStride]
  originalShape : original.shape=[rows,cols]
  resolvedShape : resolved.shape=[rows,cols]
  sameDType : original.dtype=.int8 ∧ resolved.dtype=.int8
  sameDevice : original.device=resolved.device
  byteCopy : ∀ r k,r<rows → k<cols →
    read8 prepared (resolved.storage,resolved.offset+r*resolvedDense.rowStride+k*resolvedDense.colStride)=
    read8 caller (original.storage,original.offset+r*originalDense.rowStride+k*originalDense.colStride)
  branch : (originalDense.rowStride≤1 ∨ originalDense.colStride≤1) → resolved=original

structure Invocation (s : Shape) where
  config : Config
  selected : Selection s config
  schedule : Schedule s config

def invoke (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool)
    (v : Invocation s) (m : Memory F H) :=
  launch a s v.config b v.selected divisor hasBias v.schedule m
def trials (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) :
    List (Invocation s) → Memory F H → Memory F H
  | [],m => m
  | v::vs,m => trials a s b divisor hasBias vs (invoke a s b divisor hasBias v m)
theorem trials_frame (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool)
    (vs : List (Invocation s)) (m : Memory F H) : InputsAgree b (trials a s b divisor hasBias vs m) m := by
  intro p hp
  induction vs generalizing m with
  | nil => rfl
  | cons v vs ih => rw [trials,ih]; exact launch_frame _ _ _ _ _ _ _ _ _ p hp

theorem reference_current (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers) (fresh : b.Fresh)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (q : Schedule s c)
    (m m' : Memory F H) (same : InputsAgree b m m') (x : MatrixCell s) :
    wordReference a s c b selected divisor hasBias q m x=
      wordReference a s c b selected divisor hasBias q m' x := by
  unfold wordReference
  have h : (fun z : Fin c.S => partialValue a s c b selected divisor hasBias m x.1 x.2 z)=
      (fun z : Fin c.S => partialValue a s c b selected divisor hasBias m' x.1 x.2 z) := by
    funext z
    exact partial_current a s c b fresh selected divisor hasBias m m' same _ _ _
  rw [h]

structure Returned (F H : Type) where
  tensor : Tensor
  memory : Memory F H
  currentDevice : Nat

/-- The allocated object is returned directly. Each autotune trial executes
its actual split-dependent zero hook; the selected call executes the same
hook once again. This permits any finite successful trial sequence, including
none on a cache hit. Device restoration is the actual context-manager shape. -/
def wrapper (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (allocated : Memory F H) : Returned F H :=
  if s.M=0 ∨ s.N=0 then ⟨outputTensor s b owner,allocated,callerDevice⟩
  else
    let entered := DequantizeMatmul.DeviceState.mk owner allocated
    let done := trials a s b divisor hasBias benchmarks entered.payload
    let final := invoke a s b divisor hasBias chosen done
    ⟨outputTensor s b owner,final,callerDevice⟩

theorem wrapper_returns (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory F H) :
    (wrapper a s b divisor hasBias owner callerDevice benchmarks chosen m).tensor=outputTensor s b owner ∧
    (wrapper a s b divisor hasBias owner callerDevice benchmarks chosen m).currentDevice=callerDevice := by
  unfold wrapper
  split_ifs <;> exact ⟨rfl,rfl⟩
theorem wrapper_frame (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s) (m : Memory F H) :
    InputsAgree b (wrapper a s b divisor hasBias owner callerDevice benchmarks chosen m).memory m := by
  intro p hp
  unfold wrapper
  split_ifs
  · rfl
  · change launch a s chosen.config b chosen.selected divisor hasBias chosen.schedule (trials a s b divisor hasBias benchmarks m) p=m p
    rw [launch_frame _ _ _ _ _ _ _ _ _ p hp]
    exact trials_frame _ _ _ _ _ _ _ p hp
theorem wrapper_output (a : Ops I F H) (s : Shape) (b : Buffers) (fresh : b.Fresh)
    (divisor : F) (hasBias : Bool) (owner callerDevice : Nat) (benchmarks : List (Invocation s))
    (chosen : Invocation s) (m : Memory F H) (x : MatrixCell s) :
    read16 a (wrapper a s b divisor hasBias owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      wordReference a s chosen.config b chosen.selected divisor hasBias chosen.schedule m x := by
  have nonempty : ¬(s.M=0 ∨ s.N=0) := by
    have hm := x.1.isLt
    have hn := x.2.isLt
    omega
  simp only [wrapper,if_neg nonempty,invoke]
  rw [launch_output _ _ _ _ fresh]
  exact reference_current a s chosen.config b fresh chosen.selected divisor hasBias chosen.schedule _ m
    (trials_frame a s b divisor hasBias benchmarks m) x

theorem wrapper_word (a : Ops I F H) (s : Shape) (b : Buffers) (fresh : b.Fresh)
    (divisor : F) (hasBias : Bool) (owner callerDevice : Nat) (benchmarks : List (Invocation s))
    (chosen : Invocation s) (m : Memory F H) (x : MatrixCell s) :
    (wrapper a s b divisor hasBias owner callerDevice benchmarks chosen m).memory (outputAddress s b x)=
      .inr (.inr (wordReference a s chosen.config b chosen.selected divisor hasBias chosen.schedule m x)) := by
  have nonempty : ¬(s.M=0 ∨ s.N=0) := by
    have hm := x.1.isLt
    have hn := x.2.isLt
    omega
  simp only [wrapper,if_neg nonempty,invoke]
  rw [launch_word _ _ _ _ fresh]
  rw [reference_current a s chosen.config b fresh chosen.selected divisor hasBias chosen.schedule _ m
    (trials_frame a s b divisor hasBias benchmarks m) x]

/-- Allocator/preparation boundary: newly allocated storage and per-byte
copies preserve all pre-existing caller storage. This is an ordinary allocator
contract, not a numerical obligation on the returned matrix. -/
structure AllocationABI (b : Buffers) (caller allocated : Memory F H) (callerStorage : Nat → Prop) : Prop where
  freshCaller : ¬callerStorage b.out
  frame : ∀ p,callerStorage p.1 → allocated p=caller p

theorem caller_frame (a : Ops I F H) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool)
    (owner callerDevice : Nat) (benchmarks : List (Invocation s)) (chosen : Invocation s)
    (caller allocated : Memory F H) (storage : Nat → Prop) (alloc : AllocationABI b caller allocated storage)
    (p : Address) (hp : storage p.1) :
    (wrapper a s b divisor hasBias owner callerDevice benchmarks chosen allocated).memory p=caller p := by
  have h : p.1≠b.out := by intro h; rw [h] at hp; exact alloc.freshCaller hp
  rw [wrapper_frame _ _ _ _ _ _ _ _ _ _ p h,alloc.frame p hp]

#axiomsClean trials_frame
#axiomsClean wrapper_returns
#axiomsClean wrapper_frame
#axiomsClean wrapper_output
#axiomsClean wrapper_word
#axiomsClean caller_frame
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
