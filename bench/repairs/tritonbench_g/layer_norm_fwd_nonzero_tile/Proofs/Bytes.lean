import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageExecutionReturn
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageNumerics
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ByteExecution
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormFwd
open AliasMemory
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

abbrev Bytes := Nat → UInt8

/-- One scalar's requested storage encoding. It is a local primitive; no
whole tensor, source execution, reduction or final-state equation is a field. -/
abbrev Encode (α : Type) := Key → Cell α → Nat → UInt8

def Span (a : Layout) (width : Key → Nat) (k : Key) (address : Nat) : Prop :=
  a.address k≤address ∧ address<a.address k+width k

/-- Output allocations and elements occupy fresh full byte spans. Read/read
aliases are unrestricted. This is an ordinary kernel-entry descriptor fact. -/
def Separated (a : Layout) (width : Key → Nat) (writes : Key → Prop) : Prop :=
  ∀ k, writes k → ∀ l, a.live l → l≠k → ∀ byte, byte<width l →
    ¬ Span a width k (a.address l+byte)

def Image (a : Layout) (width : Key → Nat) (encode : Encode α)
    (cells : TypedPhysical.Physical α a) (bytes : Bytes) : Prop :=
  ∀ k, a.live k → ∀ byte, byte<width k →
    bytes (a.address k+byte)=encode k (TypedPhysical.read a cells k) byte

def store (a : Layout) (width : Key → Nat) (encode : Encode α)
    (bytes : Bytes) (event : TypedPhysical.Event α) : Bytes := fun address =>
  if Span a width event.key address then
    encode event.key event.value (address-a.address event.key) else bytes address

def run (a : Layout) (width : Key → Nat) (encode : Encode α)
    (bytes : Bytes) (events : List (TypedPhysical.Event α)) : Bytes :=
  events.foldl (store a width encode) bytes

theorem store_image (a : Layout) (width : Key → Nat) (encode : Encode α)
    (writes : Key → Prop) (fresh : FreshWrites a writes) (separated : Separated a width writes)
    (cells : TypedPhysical.Physical α a) (bytes : Bytes) (image : Image a width encode cells bytes)
    (event : TypedPhysical.Event α) (written : writes event.key) :
    Image a width encode (TypedPhysical.write a cells event.key event.value)
      (store a width encode bytes event) := by
  intro key live byte bound
  by_cases same : key=event.key
  · subst key
    have hspan : Span a width event.key (a.address event.key+byte) := by unfold Span; omega
    simp [store,hspan,TypedPhysical.read,TypedPhysical.write,live]
  · have hspan := separated event.key written key live same byte bound
    have haddress : a.address key≠a.address event.key := fun h => same ((fresh event.key written).2 key live h)
    simp only [store,if_neg hspan]
    rw [image key live byte bound]
    simp [TypedPhysical.read,TypedPhysical.write,live,haddress]

theorem run_image (a : Layout) (width : Key → Nat) (encode : Encode α)
    (writes : Key → Prop) (fresh : FreshWrites a writes) (separated : Separated a width writes)
    (cells : TypedPhysical.Physical α a) (bytes : Bytes) (image : Image a width encode cells bytes)
    (events : List (TypedPhysical.Event α)) (written : ∀ event∈events, writes event.key) :
    Image a width encode (TypedPhysical.runPhysical a cells events) (run a width encode bytes events) := by
  induction events generalizing cells bytes with
  | nil => exact image
  | cons event rest ih =>
    exact ih _ _ (store_image a width encode writes fresh separated cells bytes image event
      (written event (List.mem_cons_self ..))) (fun e he => written e (List.mem_cons_of_mem _ he))

theorem run_frame (a : Layout) (width : Key → Nat) (encode : Encode α)
    (bytes : Bytes) (events : List (TypedPhysical.Event α)) (address : Nat)
    (outside : ∀ event∈events, ¬Span a width event.key address) :
    run a width encode bytes events address=bytes address := by
  induction events generalizing bytes with
  | nil => rfl
  | cons event rest ih =>
    change run a width encode (store a width encode bytes event) rest address=bytes address
    rw [ih _ (fun e he => outside e (List.mem_cons_of_mem _ he))]
    exact if_neg (outside event (List.mem_cons_self ..))

open StorageMachine

def width {α : Type} (c : Config α block) (key : Key) : Nat := (c.allocations key.1).width

def events {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (cells : Memory c) :
    List (TypedPhysical.Event α) :=
  (program c).filterMap (event c artifact (snapshot c cells))

/-- Execute all actual source-order store events on full requested byte spans.
The values are evaluated from the initial input snapshot; run_trace proves
that this equals the current-memory instruction evaluator after every store. -/
def execute {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (encode : Encode α)
    (cells : Memory c) (bytes : Bytes) : Bytes :=
  run (PhysicalDomain.layout c.allocations) (width c) encode bytes (events c artifact cells)

theorem execute_image {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (encode : Encode α)
    (separated : Separated (PhysicalDomain.layout c.allocations) (width c)
      (Writes c.flags KernelEntry.regions c.strides c.M c.N))
    (cells : Memory c) (bytes : Bytes)
    (image : Image (PhysicalDomain.layout c.allocations) (width c) encode cells bytes) :
    Image (PhysicalDomain.layout c.allocations) (width c) encode
      (StorageMachine.execute c artifact cells) (execute c artifact encode cells bytes) := by
  rw [StorageMachine.execute,StorageMachine.run_trace]
  apply run_image _ _ _ _ c.storage.fresh separated cells bytes image
  intro e he
  obtain ⟨instruction,_,he⟩ := List.mem_filterMap.mp he
  exact event_written c artifact (snapshot c cells) instruction e he

theorem execute_frame {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (encode : Encode α)
    (cells : Memory c) (bytes : Bytes) (address : Nat)
    (outside : ∀ key, Writes c.flags KernelEntry.regions c.strides c.M c.N key →
      ¬Span (PhysicalDomain.layout c.allocations) (width c) key address) :
    execute c artifact encode cells bytes address=bytes address := by
  apply run_frame
  intro e he
  obtain ⟨instruction,_,he⟩ := List.mem_filterMap.mp he
  exact outside e.key (event_written c artifact (snapshot c cells) instruction e he)

/-- A logical Tensor coordinate is observed through its actual API pointer,
strides and requested element size, with object/storage identity retained. -/
structure TensorView where
  object : Nat
  storage : Nat
  metadata : Torch.TensorMetadata
  strides : List Nat
  elementBytes : Nat
  bytesPositive : 0<elementBytes

def coordinateOffset : List Nat → List Nat → Nat
  | s::ss,i::is => s*i+coordinateOffset ss is
  | _,_ => 0

def TensorView.bytes (tensor : TensorView) (memory : Bytes) (index : List Nat) (byte : Nat) : UInt8 :=
  memory (tensor.metadata.dataPtr.toNat+tensor.elementBytes*coordinateOffset tensor.strides index+byte)

/-- These are local descriptor facts for an actual Tensor coordinate. The
same adapter supports FP32 statistics, int64 seeds, bool masks and every
accepted variable-width output dtype. It imposes no value/output equality. -/
structure Binds (a : Layout) (width : Key → Nat) (tensor : TensorView) (index : List Nat) (key : Key) : Prop where
  live : a.live key
  coordinate : List.Forall₂ (fun i n => i<n) index tensor.metadata.shape
  layout : tensor.metadata.layout=.strided
  noNegative : tensor.metadata.isNeg=Bool.false
  width : tensor.elementBytes=width key
  address : tensor.metadata.dataPtr.toNat+tensor.elementBytes*coordinateOffset tensor.strides index=a.address key

theorem tensor_readback (a : Layout) (width : Key → Nat) (encode : Encode α)
    (cells : TypedPhysical.Physical α a) (bytes : Bytes) (image : Image a width encode cells bytes)
    (tensor : TensorView) (index : List Nat) (key : Key) (binding : Binds a width tensor index key)
    (byte : Nat) (bound : byte<tensor.elementBytes) :
    tensor.bytes bytes index byte=encode key (TypedPhysical.read a cells key) byte := by
  unfold TensorView.bytes
  rw [binding.address]
  exact image key binding.live byte (by simpa [binding.width] using bound)

/-- A Tensor bound to an enabled source destination has the byte width of
that destination's requested conversion, including FP32 statistics and bool
masks. The descriptor cannot substitute a different output element width. -/
theorem tensor_requested_width {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (row : Fin c.M) (destination : Compiler.Destination) (operand : StorageCompiler.Operand)
    (enabled : artifact.stores destination=some operand)
    (tensor : TensorView) (index : List Nat)
    (binding : Binds (PhysicalDomain.layout c.allocations) (width c)
      tensor index (key c (row,destination))) :
    tensor.elementBytes=(c.profile.dtype destination).bytes := by
  exact binding.width.trans (c.outputWidths destination (store_enabled c artifact destination operand enabled).1)

#axiomsClean store_image
#axiomsClean run_image
#axiomsClean execute_image
#axiomsClean execute_frame
#axiomsClean tensor_readback
#axiomsClean tensor_requested_width
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ByteExecution
