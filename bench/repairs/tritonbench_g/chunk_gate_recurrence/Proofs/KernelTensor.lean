import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCompilerMetadata
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelIndex
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage

/-! Actual prepared kernel Tensor arguments. The seed keeps its four caller
strides, including zero strides and storage offsets. The local stride law is
ordinary Tensor address semantics; it contains no numerical result or program
execution. Readonly input aliases are unrestricted. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelTensor
open VeriTile.Triton FP.Structural KernelProgram VeriTile.Torch.FP32
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
open Classical
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
  (Placement TensorImage commit complete native_logical_output)
end Shared

def tensorShape (c : Shape) (r : RegionName) : List Nat :=
  if r="D" then [c.B,c.H,c.N]
  else if r="Seed" ∨ r="DL" then [c.B,c.H,c.K,c.V]
  else if r="DG" then [c.heads,c.N,c.KT,c.VT]
  else [c.B,c.H,c.N,c.K,c.V]
def dense (r : RegionName) : Bool := decide (r≠"Seed")
structure Buffers (c : Shape) (s : SeedStrides) (kind : Kind) (seed : Bool) where
  tensor : RegionName → Tensor
  laws : ∀ r,TensorLaws (tensor r)
  checked : ∀ r,VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract (tensor r).metadata (dense r)
  shape : ∀ r,(tensor r).metadata.shape=tensorShape c r
  device : ∀ r∈KernelExecution.inputRoles kind seed++KernelExecution.outputRoles kind,
    (tensor r).metadata.device=(tensor "S").metadata.device
  /-- Additional observed flags/strides absent from the shared metadata record. -/
  conjugate : RegionName → Bool
  noConjugate : ∀ r,conjugate r=Bool.false
  strides : RegionName → List Nat
  seedStrides : strides "Seed"=[s.batch,s.head,s.key,s.value]
  seedAddress : ∀ h k v,h<c.heads → k<c.K → v<c.V →
    (tensor "Seed").physicalIndex (seedDenseOffset c h k v)=
      (tensor "Seed").storageOffset+(h/c.H)*s.batch+(h%c.H)*s.head+k*s.key+v*s.value
  seedBacking : (tensor "Seed").storageOffset+KernelIndex.seedExtent c s≤(tensor "Seed").storage.elements

def base (b : Buffers c s kind seed) (r : RegionName) := (b.tensor r).storage.address (b.tensor r).storageOffset
def address (b : Buffers c s kind seed) (r : RegionName) (off : Nat) := base b r+4*off
def extent (b : Buffers c s kind seed) (r : RegionName) :=
  if r="Seed" then KernelIndex.seedExtent c s else (b.tensor r).metadata.numel
theorem pointer (b : Buffers c s kind seed) (r : RegionName) (nonempty : 0<(b.tensor r).metadata.numel) :
    BitVec.ofNat 64 (address b r 0)=(b.tensor r).dataPtr := by
  simpa [address,base] using (b.laws r).pointer nonempty |>.symm
theorem offset_pointer (b : Buffers c s kind seed) (r : RegionName)
    (nonempty : 0<(b.tensor r).metadata.numel) (off : Nat) :
    BitVec.ofNat 64 (address b r off)=(b.tensor r).dataPtr+BitVec.ofNat 64 (4*off) := by
  rw [←pointer b r nonempty]
  simp [address,BitVec.ofNat_add]
theorem dense_address (b : Buffers c s kind seed) (r : RegionName) (notSeed : r≠"Seed")
    (i : Nat) (hi : i<(b.tensor r).metadata.numel) :
    (b.tensor r).storage.address ((b.tensor r).physicalIndex i)=address b r i := by
  rw [(b.laws r).contiguous ((b.checked r).dense (by simp [dense,notSeed])) i hi]
  simp [Storage.address,address,base,Nat.mul_add,Nat.add_assoc]
theorem seed_address (b : Buffers c s kind seed) (p : Program c) (e : Fin 64) (j : Fin 16) :
    (b.tensor "Seed").storage.address ((b.tensor "Seed").physicalIndex (dseedAddress c p e.val j.val))=
      address b "Seed" (KernelProgram.seedAddress c p s e.val j.val) := by
  rw [KernelIndex.seed_address,b.seedAddress p.head _ _ p.headInside
    (KernelIndex.key_inside c p e) (KernelIndex.value_inside c p j)]
  simp [Storage.address,address,base,KernelProgram.seedAddress,Nat.mul_add,Nat.add_assoc]

variable {α : Type} [Inhabited α]
def physical (b : Buffers c s kind seed) (read : Nat → α) (u : State α) : State α :=
  {u with mem:=fun r off => if off<extent b r then
    (Cell.mk (KernelExecution.dtype kind r).toTileDType
      (ofFloat (KernelExecution.dtype kind r) (read (address b r off)))) else u.mem r off}
def rawCell (kind : Kind) (r : RegionName) (cell : Cell α) : α :=
  toFloat (KernelExecution.dtype kind r) (cell.read (KernelExecution.dtype kind r).toTileDType)
theorem physical_read (b : Buffers c s kind seed) (read : Nat → α) (u : State α) (r : RegionName)
    (off : Nat) (hi : off<extent b r) :
    rawCell kind r ((physical b read u).mem r off)=read (address b r off) := by
  simp only [physical,hi,ite_true,rawCell,Cell.read_same]
  cases KernelExecution.dtype kind r <;> rfl
theorem dense_read (b : Buffers c s kind seed) (read : Nat → VeriTile.Triton.FP.Value .fp32)
    (r : RegionName) (notSeed : r≠"Seed") (i : Nat) (hi : i<(b.tensor r).metadata.numel) :
    read (address b r i)=logicalValue read (b.tensor r) (b.laws r) i := by
  simp only [logicalValue,(b.checked r).noNeg,Bool.false_eq_true,if_false]
  rw [dense_address b r notSeed i hi]
theorem seed_read (b : Buffers c s kind seed) (read : Nat → VeriTile.Triton.FP.Value .fp32)
    (p : Program c) (e : Fin 64) (j : Fin 16) :
    read (address b "Seed" (KernelProgram.seedAddress c p s e.val j.val))=
      logicalValue read (b.tensor "Seed") (b.laws "Seed") (dseedAddress c p e.val j.val) := by
  simp only [logicalValue,(b.checked "Seed").noNeg,Bool.false_eq_true,if_false]
  rw [seed_address b p e j]

def Pointers (platform : KernelCompiler.Platform) (binding : KernelCompiler.Binding platform kind c s seed)
    (b : Buffers c s kind seed) : Prop :=
  ∀ r∈KernelExecution.inputRoles kind seed++KernelExecution.outputRoles kind,
    0<binding.object.signature.pointerDivisibility r ∧
    (b.tensor r).dataPtr.toNat%binding.object.signature.pointerDivisibility r=0

structure Outputs (b : Buffers c s kind seed) (p : Program c) extends
    Shared.Placement (KernelMemory.footprint kind c p) where
  same : ∀ r∈KernelExecution.outputRoles kind,tensor r=b.tensor r
  /-- Ordinary output-buffer address separation also covers unwritten slices. -/
  separate : ∀ r i,KernelMemory.footprint kind c p r i →
    ∀ q∈KernelExecution.outputRoles kind,∀ j,j<(b.tensor q).metadata.numel → (r,i)≠(q,j) →
      Memory.Apart ((dataView (tensor r) (laws r) ((checked r).dense rfl)).address i)
        ((b.tensor q).storage.address ((b.tensor q).physicalIndex j))
def Isolated (b : Buffers c s kind seed) (p : Program c) (out : Outputs b p) : Prop :=
  ∀ r i,KernelMemory.footprint kind c p r i →
    ∀ q∈KernelExecution.inputRoles kind seed,∀ j,j<extent b q →
      Memory.Apart (out.toPlacement.address r i) (address b q j)

def observe (kind : Kind) (r : RegionName) (_off : Nat) (cell : Cell KernelCarrier.Node) : KernelCarrier.Word :=
  (rawCell kind r cell).value
theorem output {Store Byte : Type} (api : Memory.API KernelCarrier.Word Store Byte)
    (b : Buffers c s kind seed) (p : Program c) (out : Outputs b p)
    (v : State KernelCarrier.Node) (initial : Store) (r : RegionName)
    (hr : r∈KernelExecution.outputRoles kind) (off : Nat) (active : KernelMemory.footprint kind c p r off) :
    api.read (Shared.commit api out.toPlacement (observe kind) v initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex off))=observe kind r off (v.mem r off) := by
  rw [←out.same r hr]
  exact VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.commit_output api out.toPlacement _ v initial r off active
theorem input_frame {Store Byte : Type} (api : Memory.API KernelCarrier.Word Store Byte)
    (b : Buffers c s kind seed) (p : Program c) (out : Outputs b p) (isolated : Isolated b p out)
    (v : State KernelCarrier.Node) (initial : Store) :
    ∀ r∈KernelExecution.inputRoles kind seed,∀ i,i<extent b r →
      api.read (Shared.commit api out.toPlacement (observe kind) v initial) (address b r i)=
        api.read initial (address b r i) := by
  intro r hr i hi
  apply Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  exact isolated a.1 a.2 ((out.coverage a.1 a.2).mp ha) r hr i hi

theorem unwritten_output_frame {Store Byte : Type} (api : Memory.API KernelCarrier.Word Store Byte)
    (b : Buffers c s kind seed) (p : Program c) (out : Outputs b p)
    (v : State KernelCarrier.Node) (initial : Store) (r : RegionName)
    (hr : r∈KernelExecution.outputRoles kind) (i : Nat) (hi : i<(b.tensor r).metadata.numel)
    (outside : ¬KernelMemory.footprint kind c p r i) :
    api.read (Shared.commit api out.toPlacement (observe kind) v initial)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))=
      api.read initial ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)) := by
  apply Memory.run_read_frame
  intro e he
  obtain ⟨a,ha,rfl⟩ := List.mem_map.mp he
  have active := (out.coverage a.1 a.2).mp ha
  apply out.separate a.1 a.2 active r hr i hi
  intro equal
  have first := congrArg Prod.fst equal
  have second := congrArg Prod.snd equal
  change a.1=r at first
  change a.2=i at second
  exact outside (by simpa only [first,second] using active)

#axiomsClean pointer
#axiomsClean seed_address
#axiomsClean physical_read
#axiomsClean output
#axiomsClean input_frame
#axiomsClean unwritten_output_frame
#specNonCircular Buffers avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelTensor
