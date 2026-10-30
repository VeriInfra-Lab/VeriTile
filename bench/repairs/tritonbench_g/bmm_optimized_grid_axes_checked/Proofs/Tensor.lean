import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Writes

/-! Ordinary Torch API/storage adapter. Inputs share one allocation-indexed
physical reader, so zero strides, overlapping views and storage offsets are
retained. The existing contiguous calls materialize Host.prepare; allocator and
copy implementations remain API boundaries, not new numerical assumptions. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.TensorAPI
open VeriTile.Triton FP.Structural VeriTile.Torch
open BmmOptimized
noncomputable section
open scoped BigOperators

structure Tensor where
  object : Nat
  metadata : TensorMetadata
  storageId : Nat
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  batchStride : Nat
  rowStride : Nat
  columnStride : Nat

def Tensor.view (t : Tensor) : Host.View :=
  ⟨t.storageId,t.metadata.device.index,t.batchStride,t.rowStride,t.columnStride,
    t.storageElements-t.storageOffset,t.metadata.isNeg⟩

def Tensor.wordAddress (t : Tensor) (b row col : Nat) :=
  t.storageOffset+b*t.batchStride+row*t.rowStride+col*t.columnStride

structure Laws (t : Tensor) (batch rows cols : Nat) : Prop where
  shape : t.metadata.shape=[batch,rows,cols]
  dtype : t.metadata.dtype=.floating .fp32
  layout : t.metadata.layout=.strided
  cuda : t.metadata.device.kind="cuda"
  pointer : 0<batch*rows*cols → t.metadata.dataPtr.toNat=t.storageBase+4*t.storageOffset
  span : 0<batch → 0<rows → 0<cols →
    (batch-1)*t.batchStride+(rows-1)*t.rowStride+(cols-1)*t.columnStride<
      t.storageElements-t.storageOffset

structure Invocation where
  shape : Shape
  left : Tensor
  right : Tensor
  outputObject : Nat
  outputStorage : Nat
  outputBase : Nat
  callerDevice : Nat

def Invocation.call (a : Invocation) : Host.Call :=
  ⟨a.shape,a.left.view,a.right.view,a.outputStorage⟩
def Invocation.metadata (a : Invocation) : Checked.Metadata :=
  ⟨a.call,a.left.metadata.dataPtr.toNat,a.right.metadata.dataPtr.toNat⟩

structure Ordinary (a : Invocation) : Prop where
  left : Laws a.left a.shape.batch a.shape.m a.shape.k
  right : Laws a.right a.shape.batch a.shape.k a.shape.n
  freshLeft : a.outputStorage≠a.left.storageId
  freshRight : a.outputStorage≠a.right.storageId
  outputPointer : a.outputBase<2^64

theorem ordinary (a : Invocation) (h : Ordinary a) : Checked.Ordinary a.call :=
  ⟨h.left.span,h.right.span,h.freshLeft,h.freshRight⟩

def returned (a : Invocation) : Tensor :=
  ⟨a.outputObject,
    ⟨.floating .fp32,.strided,⟨"cuda",a.left.metadata.device.index⟩,
      [a.shape.batch,a.shape.m,a.shape.n],Bool.true,Bool.false,BitVec.ofNat 64 a.outputBase⟩,
    a.outputStorage,a.outputBase,a.shape.size,0,
    max a.shape.m 1*max a.shape.n 1,max a.shape.n 1,1⟩

abbrev Physical (α : Type) := Nat → Nat → α

def read {α : Type} (memory : Physical α) (t : Tensor) (b row col : Nat) : α :=
  memory t.storageId (t.wordAddress b row col)

/-- Caller projections use the same allocation map, including aliased inputs. -/
def caller {α : Type} (a : Invocation) (memory : Physical α) : Memory α :=
  fun region off=>match region with
    | .callerA=>memory a.left.storageId (a.left.storageOffset+off)
    | .callerB=>memory a.right.storageId (a.right.storageOffset+off)
    | .output=>memory a.outputStorage off
    | .a=>memory a.left.storageId (a.left.storageOffset+off)
    | .b=>memory a.right.storageId (a.right.storageOffset+off)
    | .other id=>memory id off

/-- The concrete tag/state loaded by the source interpreter. Exterior cells
are retained as cells of any dtype. Input staging is exactly the existing
contiguous API's logical preparation, proved against original view addresses. -/
def initial {α : Type} [Inhabited α] (a : Invocation) (memory : Physical α)
    (exterior : RegionName → Nat → FP.Structural.Cell α) : State α :=
  let m:=Host.prepare id a.call (caller a memory)
  {mem:=fun r off=>if r="A" then .mk .real (m .a off)
    else if r="B" then .mk .real (m .b off)
    else if r="O" then .mk .real (m .output off) else exterior r off
   regs:=fun _ _ _=>none
   pids:=fun _=>0
   numPids:=fun _=>1
   undef:=fun dtype _ _=>defaultValue dtype}

theorem initial_related {α : Type} [Inhabited α] (a : Invocation) (memory : Physical α)
    (exterior : RegionName → Nat → FP.Structural.Cell α) :
    Bridge.Related (Host.prepare id a.call (caller a memory)) (initial a memory exterior) := by
  constructor <;> intro off <;> simp [initial]

theorem caller_left {α : Type} (a : Invocation) (memory : Physical α) (b row col : Nat) :
    caller a memory .callerA (a.call.a.address b row col)=read memory a.left b row col := by
  simp [caller,read,Invocation.call,Tensor.view,Host.View.address,Tensor.wordAddress,Nat.add_assoc]
theorem caller_right {α : Type} (a : Invocation) (memory : Physical α) (b row col : Nat) :
    caller a memory .callerB (a.call.b.address b row col)=read memory a.right b row col := by
  simp [caller,read,Invocation.call,Tensor.view,Host.View.address,Tensor.wordAddress,Nat.add_assoc]

/-- Independent ordinary Tensor mathematical result. -/
def product (a : Invocation) (memory : Physical ℝ) (x : BmmOptimized.Cell) : ℝ :=
  ∑ k∈Finset.range a.shape.k,
    read memory a.left x.batch x.row k*read memory a.right x.batch k x.column

/-- Independent native padded scalar instruction chain. -/
def chain (ops : Native.Instructions) (a : Invocation) (memory : Physical Native.Word)
    (x : BmmOptimized.Cell) : Nat → Native.Word
  | 0=>Native.zero
  | k+1=>ops.fmaRN
      (if k<a.shape.k then read memory a.left x.batch x.row k else Native.zero)
      (if k<a.shape.k then read memory a.right x.batch k x.column else Native.zero)
      (chain ops a memory x k)

theorem product_host (a : Invocation) (admitted : InputContract a.metadata)
    (memory : Physical ℝ) (x : BmmOptimized.Cell) :
    Host.product a.call (caller a memory) x=product a memory x := by
  have raw:=Checked.raw a.metadata admitted.original
  change a.call.a.negative=Bool.false ∧ a.call.b.negative=Bool.false at raw
  simp only [Host.product,product,Host.logical,raw.1,raw.2,Bool.false_eq_true,ite_false,
    caller_left,caller_right]
  rfl

theorem chain_host (ops : Native.Instructions) (a : Invocation)
    (admitted : InputContract a.metadata) (memory : Physical Native.Word)
    (x : BmmOptimized.Cell) (n : Nat) :
    Host.chain ops a.call (caller a memory) x n=chain ops a memory x n := by
  have raw:=Checked.raw a.metadata admitted.original
  change a.call.a.negative=Bool.false ∧ a.call.b.negative=Bool.false at raw
  induction n with
  | zero=>rfl
  | succ n ih=>
    rw [Host.chain,chain,ih]
    simp only [Host.logical,raw.1,raw.2,Bool.false_eq_true,ite_false,caller_left,caller_right]
    rfl

/-- Rebind the source's output region to the fresh physical allocation. This
is word-addressed readback of the source stores, not a mathematical oracle. -/
def publish {α : Type} [Inhabited α] (a : Invocation) (before : Physical α)
    (final : State α) : Physical α :=
  fun allocation off=>if allocation=a.outputStorage ∧ off<a.shape.size then
    (final.mem "O" off).read .real else before allocation off

theorem publish_output {α : Type} [Inhabited α] (a : Invocation) (before : Physical α)
    (final : State α) (off : Nat) (h : off<a.shape.size) :
    publish a before final a.outputStorage off=(final.mem "O" off).read .real := by
  simp [publish,h]
theorem publish_frame {α : Type} [Inhabited α] (a : Invocation) (before : Physical α)
    (final : State α) (allocation off : Nat)
    (outside : allocation≠a.outputStorage ∨ a.shape.size≤off) :
    publish a before final allocation off=before allocation off := by
  rcases outside with h|h
  · simp [publish,h]
  · simp [publish,show ¬off<a.shape.size from by omega]

theorem returned_address (a : Invocation) (x : BmmOptimized.Cell) (hx : a.shape.Active x) :
    (returned a).wordAddress x.batch x.row x.column=a.shape.o x.batch x.row x.column := by
  have hm : 1≤a.shape.m := by have :=hx.2.1;omega
  have hn : 1≤a.shape.n := by have :=hx.2.2;omega
  simp [returned,Tensor.wordAddress,Shape.o,Nat.max_eq_left hm,Nat.max_eq_left hn]
  ring

theorem read_returned {α : Type} [Inhabited α] (a : Invocation) (before : Physical α)
    (final : State α) (x : BmmOptimized.Cell) (hx : a.shape.Active x) :
    read (publish a before final) (returned a) x.batch x.row x.column=
      (final.mem "O" (a.shape.o x.batch x.row x.column)).read .real := by
  rw [read,returned_address a x hx]
  exact publish_output a before final _ (address_bound a.shape x hx)

theorem real_target (a : Invocation) (admitted : InputContract a.metadata)
    (memory : Physical ℝ) (x : BmmOptimized.Cell) (hx : a.shape.Active x) :
    ordered a.shape realScalar (Host.prepare id a.call (caller a memory)) x=product a memory x := by
  have prep:=Checked.prepare_raw a.metadata admitted.original Neg.neg (caller a memory)
  change Host.prepare Neg.neg a.call (caller a memory)=Host.prepare id a.call (caller a memory) at prep
  rw [←prep,ordered_real]
  exact (Host.prepared_product a.call (caller a memory) x hx).trans (product_host a admitted memory x)

theorem native_target (ops : Native.Instructions) (a : Invocation) (admitted : InputContract a.metadata)
    (memory : Physical Native.Word) (x : BmmOptimized.Cell) (hx : a.shape.Active x) :
    ordered a.shape (Native.scalar ops) (Host.prepare id a.call (caller a memory)) x=
      chain ops a memory x (a.shape.iterations*32) := by
  have prep:=Checked.prepare_raw a.metadata admitted.original ops.neg (caller a memory)
  change Host.prepare ops.neg a.call (caller a memory)=Host.prepare id a.call (caller a memory) at prep
  rw [←prep]
  exact (Host.prepared_chain ops a.call (caller a memory) x hx _).trans (chain_host ops a admitted memory x _)

theorem chain_equiv (actual target : Native.Instructions)
    (fma : ∀ a b carry,actual.fmaRN a b carry=target.fmaRN a b carry)
    (a : Invocation) (memory : Physical Native.Word) (x : BmmOptimized.Cell) (n : Nat) :
    chain actual a memory x n=chain target a memory x n := by
  induction n with
  | zero=>rfl
  | succ n ih=>rw [chain,chain,ih,fma]

abbrev PhysicalBytes := Physical Native.Byte

def loadWords (memory : PhysicalBytes) : Physical Native.Word :=
  fun allocation off i=>memory allocation (4*off+i.val)

def publishBytes (a : Invocation) (before : PhysicalBytes) (final : State Native.Word) : PhysicalBytes :=
  fun allocation off=>if allocation=a.outputStorage ∧ off<4*a.shape.size then
    ((final.mem "O" (off/4)).read .real) ⟨off%4,Nat.mod_lt _ (by decide)⟩ else before allocation off

def tensorByte (memory : PhysicalBytes) (t : Tensor) (b row col : Nat) (i : Fin 4) :=
  memory t.storageId (4*t.wordAddress b row col+i.val)

theorem bytes_output (a : Invocation) (before : PhysicalBytes) (final : State Native.Word)
    (x : BmmOptimized.Cell) (hx : a.shape.Active x) (i : Fin 4) :
    tensorByte (publishBytes a before final) (returned a) x.batch x.row x.column i=
      ((final.mem "O" (a.shape.o x.batch x.row x.column)).read .real) i := by
  unfold tensorByte
  rw [returned_address a x hx]
  have bound:=address_bound a.shape x hx
  have hd : (4*a.shape.o x.batch x.row x.column+i.val)/4=a.shape.o x.batch x.row x.column := by omega
  have hm : (4*a.shape.o x.batch x.row x.column+i.val)%4=i.val := by omega
  simp only [publishBytes,returned,hd]
  rw [if_pos (by simp only [true_and];omega)]
  exact congrArg ((final.mem "O" (a.shape.o x.batch x.row x.column)).read .real) (Fin.ext hm)

theorem bytes_frame (a : Invocation) (before : PhysicalBytes) (final : State Native.Word)
    (allocation off : Nat) (outside : allocation≠a.outputStorage ∨ 4*a.shape.size≤off) :
    publishBytes a before final allocation off=before allocation off := by
  rcases outside with h|h
  · simp [publishBytes,h]
  · simp [publishBytes,show ¬off<4*a.shape.size from by omega]

#specNonCircular product avoiding [initial,publish,Bridge.sourceWrites,Execution.exec]
#specNonCircular chain avoiding [initial,publish,Bridge.sourceWrites,Execution.exec]
#axiomsClean initial_related
#axiomsClean product_host
#axiomsClean chain_host
#axiomsClean read_returned
#axiomsClean publish_frame
#axiomsClean real_target
#axiomsClean native_target
#axiomsClean chain_equiv
#axiomsClean bytes_output
#axiomsClean bytes_frame
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.TensorAPI
