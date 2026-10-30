import bench.repairs.tritonbench_g.triton_argmax.Proofs.DimMemory
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Real
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Formats

/-! Host contracts are ordinary tensor/allocation and per-device current-stream
primitives. Views retain storage offsets, strides and lazy negation. Returned
squeeze views keep the actual allocation. Setup rejection is an explicit API
outcome; the successful path does not assume a whole-kernel result. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Host
open Source Memory

structure Tensor where
  object : Nat
  dtype : String
  storage : Allocation
  shape : List Nat
  strides : List Nat
  storageOffset : Nat
  lazyNegative : Bool
  /-- Actual host-library scalar negation, kept explicit for NaN payloads and
  signed-zero behavior. It is a scalar primitive, never an output oracle. -/
  negate32 : Word → Word

def Tensor.numel (t : Tensor) : Nat := t.shape.prod

def viewOffset : List Nat → List Nat → Nat → Nat
  | _::tail,stride::rest,i => (i/tail.prod)*stride+viewOffset tail rest (i%tail.prod)
  | _,_,_ => 0

def Tensor.offset (t : Tensor) (i : Nat) : Nat := t.storageOffset+viewOffset t.shape t.strides i

/-- A contiguous view can retain a nonzero storage offset. Kernel data_ptr
starts at this rebased window, not at the backing storage's first byte. -/
def Tensor.dataWindow (t : Tensor) : Allocation :=
  ⟨t.storage.owner,t.storage.base+4*t.storageOffset,t.storage.bytes-4*t.storageOffset⟩

def logicalWord {M : Type} (mem : Primitives M) (s : M) (t : Tensor) (i : Nat) : Word :=
  let physical := mem.read 4 s (address t.storage 4 (t.offset i))
  if t.lazyNegative then t.negate32 physical else physical

def ValidView (t : Tensor) : Prop :=
  t.strides.length=t.shape.length ∧ ∀ i,i < t.numel → 4*(t.offset i+1) ≤ t.storage.bytes

/-- Ordinary coordinate-preserving library preparation contract.
Broadcast and overlapping logical coordinates are allowed. Input storage is
read-only; a returned packed allocation may alias a previously packed input. -/
structure Packed {M : Type} (mem : Primitives M) (before : M) (original : Tensor) where
  tensor : Tensor
  state : M
  shape : tensor.shape=original.shape
  device : tensor.storage.owner=original.storage.owner
  capacity : 4*original.numel ≤ tensor.dataWindow.bytes
  coordinates : ∀ i,i < original.numel →
    mem.read 4 state (address tensor.dataWindow 4 i)=logicalWord mem before original i
  callerFrame : ∀ p,original.storage.contains p → mem.byte state p=mem.byte before p
  libraryWrites : List Allocation
  libraryFrame : ∀ p,(∀ a,a∈libraryWrites → ¬a.contains p) → mem.byte state p=mem.byte before p

inductive DTypeToken where
  | omitted
  | supported : MidFormat → DTypeToken
  | unsupported : Nat → DTypeToken

structure Request where
  input : Tensor
  dim : Option Int
  keepdim : Bool
  dtype : DTypeToken

def normalizeDim (rank : Nat) (dim : Int) : Option Nat :=
  if -(Int.ofNat rank) ≤ dim ∧ dim < Int.ofNat rank then
    some ((dim%Int.ofNat rank).toNat)
  else none

def outputShape (shape : List Nat) (dim : Option Nat) (keep : Bool) : List Nat :=
  match dim with
  | none => if keep then shape.map (fun _ => 1) else []
  | some ax => if keep then shape.set ax 1 else shape.eraseIdx ax

inductive Dispatch where
  | global : Nat → List Nat → Dispatch
  | dimension : Nat → Nat → Nat → Nat → List Nat → Dispatch
  | emptyDimension : Nat → List Nat → Dispatch
  | invalidDimension
  | emptyReduction
  deriving DecidableEq,Repr

/-- Exact source branch order: invalid dimension precedes allocation; an empty
reduction rejects, while a nonempty axis with zero outer/inner extent returns
the allocated/squeezed empty result before launch. -/
def dispatch (r : Request) : Dispatch :=
  match r.dim with
  | none => if r.input.numel=0 then .emptyReduction
      else .global r.input.numel (outputShape r.input.shape none r.keepdim)
  | some d => match normalizeDim r.input.shape.length d with
      | none => .invalidDimension
      | some ax =>
          let n := r.input.shape[ax]?.getD 0
          let m := (r.input.shape.take ax).prod
          let k := (r.input.shape.drop (ax+1)).prod
          if n=0 then .emptyReduction
          else if m=0 ∨ k=0 then .emptyDimension ax (outputShape r.input.shape (some ax) r.keepdim)
          else .dimension ax m n k (outputShape r.input.shape (some ax) r.keepdim)

/-- The actual allocated object and optional squeeze view. The returned object
identity may differ after squeeze, but its allocation and byte origin do not. -/
structure Output where
  allocated : Tensor
  returned : Tensor
  sameStorage : returned.storage=allocated.storage
  sameOffset : returned.storageOffset=allocated.storageOffset
  zeroOffset : allocated.storageOffset=0
  int64 : returned.dtype="torch.int64"
  linearCoordinates : ∀ i,i < returned.numel → returned.offset i=i

def outputWord {M : Type} (mem : Primitives M) (s : M) (out : Output) (i : Nat) : BitVec 64 :=
  mem.read 8 s (address out.returned.storage 8 (out.returned.offset i))

/-- Allocated output/scratch capacity and native admission come from ordinary
allocator/frontend metadata. They are not numerical assumptions. -/
structure AllocationFacts (a : Allocation) where
  nativeSize : a.bytes < Native.signedLimit

inductive SetupError where
  | invalidDim | emptyReduction | allocation | unsupportedDtype | frontend | launch
  deriving DecidableEq,Repr

inductive Outcome (M : Type) where
  | rejected : SetupError → M → Outcome M
  | returned : Output → M → Outcome M

abbrev Stream := Device × Nat

structure Context where
  currentDevice : Device
  currentStream : Device → Nat

def streamFor (ctx : Context) (device : Device) : Stream := (device,ctx.currentStream device)

/-- Device context selection retains each device's current stream. -/
def withDevice (ctx : Context) (device : Device) : Context := { ctx with currentDevice := device }

theorem with_device_stream (ctx : Context) (device : Device) :
    streamFor (withDevice ctx device) device=streamFor ctx device := rfl

structure Command (M : Type) where
  stream : Stream
  action : M → M

/-- Ordinary FIFO completion on one current stream. This is the model's
observation after queued dependencies, not an invented device-wide barrier. -/
def complete {M : Type} (stream : Stream) (commands : List (Command M)) (s : M) : M :=
  ((commands.filter fun command => command.stream=stream).map Command.action).foldl (fun s f => f s) s

theorem complete_append {M : Type} (stream : Stream) (a b : List (Command M)) (s : M) :
    complete stream (a++b) s=complete stream b (complete stream a s) := by
  simp only [complete,List.filter_append,List.map_append,List.foldl_append]

def initializeIndices {M : Type} (mem : Primitives M) (p : GlobalMemory.Buffers)
    (count : Nat) (s : M) : M := stores mem p.midIndex 8 id (fun _ => 0) (List.range count) s

theorem initialized {M : Type} (mem : Primitives M) (p : GlobalMemory.Buffers)
    (count : Nat) (s : M) (i : Nat) (valid : i < count) :
    mem.read 8 (initializeIndices mem p count s) (address p.midIndex 8 i)=0 := by
  apply readback mem p.midIndex 8 (by omega) id (fun _ => 0) (List.range count) s i
  · exact List.mem_range.mpr valid
  · intros; rfl

def globalCommands {M : Type} {d : MidFormat} (ctx : Context) (device : Device)
    (mem : Primitives M) (ops : Scalars d) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (resolveContiguous : M → M) : List (Command M) :=
  let stream := streamFor ctx device
  [⟨stream,resolveContiguous⟩,⟨stream,initializeIndices mem p c.midBlock⟩,
   ⟨stream,GlobalMemory.firstGrid mem ops c p⟩,⟨stream,GlobalMemory.secondProgram mem ops c p⟩]

theorem global_fifo {M : Type} {d : MidFormat} (ctx : Context) (device : Device)
    (mem : Primitives M) (ops : Scalars d) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (resolveContiguous : M → M) (s : M) :
    complete (streamFor ctx device) (globalCommands ctx device mem ops c p resolveContiguous) s=
      GlobalMemory.run mem ops c p (initializeIndices mem p c.midBlock (resolveContiguous s)) := by
  simp [complete,globalCommands,GlobalMemory.run]

def dimCommands {M : Type} {d : MidFormat} (ctx : Context) (device : Device)
    (mem : Primitives M) (ops : Scalars d) (c : DimConfig) (p : DimMemory.Buffers)
    (resolveContiguous : M → M) : List (Command M) :=
  let stream := streamFor ctx device
  [⟨stream,resolveContiguous⟩,⟨stream,DimMemory.run mem ops c p⟩]

theorem dimension_fifo {M : Type} {d : MidFormat} (ctx : Context) (device : Device)
    (mem : Primitives M) (ops : Scalars d) (c : DimConfig) (p : DimMemory.Buffers)
    (resolveContiguous : M → M) (s : M) :
    complete (streamFor ctx device) (dimCommands ctx device mem ops c p resolveContiguous) s=
      DimMemory.run mem ops c p (resolveContiguous s) := by
  simp [complete,dimCommands]

/-- cdiv is exact integer arithmetic. The floating host sqrt/ceil/next-power
policy supplies the positive actual block, without replacing it by Nat.sqrt. -/
theorem cdiv_coverage (size block : Nat) (hs : 0 < size) (hb : 0 < block) :
    let count := (size+block-1)/block
    0 < count ∧ size ≤ count*block ∧ ∀ pid,pid < count → pid*block < size := by
  have division := Nat.div_add_mod (size+block-1) block
  rw [Nat.mul_comm block] at division
  have remainder := Nat.mod_lt (size+block-1) hb
  have subtract : size+block-1+1=size+block := by omega
  have bound : (size+block-1)/block*block ≤ size+block-1 := by
    omega
  have cover : size ≤ (size+block-1)/block*block := by omega
  dsimp only
  have countPositive : 0 < (size+block-1)/block := by
    by_cases zero : ((size+block-1)/block : Nat)=0
    · simp only [zero,Nat.zero_mul] at cover
      omega
    · exact Nat.pos_of_ne_zero zero
  refine ⟨countPositive,cover,?_⟩
  intro pid hp
  have next := Nat.mul_le_mul_right block (show pid+1 ≤ (size+block-1)/block by omega)
  simp only [Nat.add_mul,Nat.one_mul] at next
  omega

theorem global_coverage (c : GlobalConfig) (positive : 0 < c.size) (blockPositive : 0 < c.block)
    (ceilBlocks : c.blocks=(c.size+c.block-1)/c.block) (midFits : c.blocks ≤ c.midBlock)
    (firstComplete : ∀ pid,pid < c.blocks → (c.firstTree pid).Complete)
    (secondComplete : c.secondTree.Complete) : Math.GlobalCoverage c := by
  obtain ⟨blocks,cover,each⟩ := cdiv_coverage c.size c.block positive blockPositive
  exact ⟨positive,blockPositive,ceilBlocks ▸ blocks,ceilBlocks ▸ cover,
    fun pid hp => each pid (ceilBlocks ▸ hp),midFits,firstComplete,secondComplete⟩

theorem native_input_bound (size : Nat) (input : Allocation)
    (capacity : 4*size ≤ input.bytes) (metadata : AllocationFacts input) :
    4*size < Native.signedLimit := capacity.trans_lt metadata.nativeSize

#axiomsClean with_device_stream
#axiomsClean complete_append
#axiomsClean initialized
#axiomsClean global_fifo
#axiomsClean dimension_fifo
#axiomsClean cdiv_coverage
#axiomsClean global_coverage
#axiomsClean native_input_bound
end VeriTile.Bench.Repairs.TritonArgmax.Host
