import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelGeometry
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage
/-! Per-program interpretation of the actual shared FP32 Tensor arguments.
The finite commit contains exactly the source's stores. Its coverage, bounds
and separation follow from the input contract; they are not output premises. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelTensor
open VeriTile.Triton FP.Structural KernelProgram VeriTile.Torch.FP32
open VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
  (Placement TensorImage commit complete native_logical_output)
end Shared
noncomputable section
open Classical
set_option maxHeartbeats 12000000

structure Buffers (kind : Kind) (s : Shape) (f : Flags) where
  tensor : RegionName → Tensor
  laws : ∀ r,TensorLaws (tensor r)
  checked : ∀ r,VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract (tensor r).metadata Bool.true
  strides : Region → List Nat
  isConj : Region → Bool
  admitted : KernelContract.InputContract kind s f
    (fun r => ⟨(tensor (region r)).metadata,strides r,isConj r⟩)

def Buffers.view (b : Buffers kind s f) (r : RegionName) : DataView :=
  dataView (b.tensor r) (b.laws r) ((b.checked r).dense rfl)
theorem Buffers.count (b : Buffers kind s f) (r : Region) (hr : r∈KernelContract.roles kind f) :
    (b.tensor (region r)).metadata.numel=Geometry.capacity s f r := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [(b.admitted.tensors r hr).shape,KernelContract.capacity]
theorem Buffers.pointer (b : Buffers kind s f) (r : RegionName) (hn : 0<(b.tensor r).metadata.numel) :
    BitVec.ofNat 64 ((b.view r).address 0)=(b.tensor r).dataPtr :=
  actual_pointer _ _ _ hn
theorem Buffers.base (b : Buffers kind s f) (r : RegionName) (hn : 0<(b.tensor r).metadata.numel) :
    (b.view r).base=(b.tensor r).metadata.dataPtr.toNat := by
  have eq := congrArg BitVec.toNat (b.pointer r hn)
  have span := (b.view r).spanFits
  have size : (b.view r).elements=(b.tensor r).metadata.numel := rfl
  have bound : (b.view r).base<2^64 := by rw [size] at span; omega
  simpa only [DataView.address,Nat.mul_zero,Nat.add_zero,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using eq

variable {α : Type} [Inhabited α]
def physical (b : Buffers kind s f) (read : RegionName → Nat → Nat → α) (u : State α) : State α :=
  {u with mem:=fun r i => if i<(b.tensor r).metadata.numel then
    Cell.mk .real (read r i ((b.view r).address i)) else u.mem r i}
def logical (b : Buffers kind s f) (read : RegionName → Nat → Nat → α) (u : State α) : State α :=
  {u with mem:=fun r i => if i<(b.tensor r).metadata.numel then
    Cell.mk .real (read r i ((b.tensor r).storage.address ((b.tensor r).physicalIndex i))) else u.mem r i}
theorem logical_address (b : Buffers kind s f) (r : RegionName) (i : Nat)
    (hi : i<(b.tensor r).metadata.numel) :
    (b.tensor r).storage.address ((b.tensor r).physicalIndex i)=(b.view r).address i := by
  rw [(b.laws r).contiguous ((b.checked r).dense rfl) i hi]
  simp only [Buffers.view,dataView,DataView.address,DataView.base,
    if_neg (by omega : (b.tensor r).metadata.numel≠0),Storage.address]
  omega
theorem physical_eq_logical (b : Buffers kind s f) (read : RegionName → Nat → Nat → α) (u : State α) :
    physical b read u=logical b read u := by
  have mem : (physical b read u).mem=(logical b read u).mem := by
    funext r i
    by_cases hi : i<(b.tensor r).metadata.numel
    · simp [physical,logical,hi,logical_address b r i hi]
    · simp [physical,logical,hi]
  exact congrArg (fun m => ({u with mem:=m} : State α)) mem

theorem native_input (b : Buffers kind s f) (read : Nat → KernelNativeCarrier.Word)
    (u : State KernelNativeCarrier.Node) (r : RegionName) (i : Nat) (hi : i<(b.tensor r).metadata.numel) :
    ((physical b (fun r i a => KernelNativeCarrier.Node.atom (.input r i) (read a)) u).mem r i).read .real=
      KernelNativeCarrier.Node.atom (.input r i) ((logicalValue (fun a => (⟨read a⟩ : Word)) (b.tensor r) (b.laws r) i).bits) := by
  simp only [physical,hi,ite_true,Cell.read_same,logicalValue,(b.checked r).noNeg,Bool.false_eq_true,if_false]
  rw [logical_address b r i hi]

def cells (kind : Kind) (s : Shape) (f : Flags) (p : Program kind s) : List (RegionName × Nat) :=
  (((KernelContract.outputRoles kind f).flatMap fun r =>
    (List.range (Geometry.capacity s f r)).map fun i => (region r,i)).filter
      fun a => decide (KernelGeometry.Writes kind s f p a.1 a.2)).dedup
theorem cells_coverage (kind : Kind) (s : Shape) (f : Flags) (p : Program kind s) (r : RegionName) (i : Nat) :
    (r,i)∈cells kind s f p ↔ KernelGeometry.Writes kind s f p r i := by
  simp only [cells,List.mem_dedup,List.mem_filter,decide_eq_true_eq]
  constructor
  · exact And.right
  · intro active
    refine ⟨?_,active⟩
    obtain ⟨q,hq,eq,bound⟩ := KernelGeometry.output_bounds kind s f p r i active
    apply List.mem_flatMap.mpr
    exact ⟨q,hq,List.mem_map.mpr ⟨i,List.mem_range.mpr bound,by simp [eq]⟩⟩
theorem output_bound (b : Buffers kind s f) (p : Program kind s) (r : RegionName) (i : Nat)
    (active : KernelGeometry.Writes kind s f p r i) : i<(b.tensor r).metadata.numel := by
  obtain ⟨q,hq,eq,bound⟩ := KernelGeometry.output_bounds kind s f p r i active
  rw [←eq,b.count q (KernelGeometry.outputs_are_arguments kind f q hq)]
  exact bound

theorem distinct_roles_apart (b : Buffers kind s f) (r r' : Region)
    (hr : r∈KernelContract.roles kind f) (hr' : r'∈KernelContract.roles kind f) (different : r≠r')
    (i j : Nat) (hi : i<(b.tensor (region r)).metadata.numel) (hj : j<(b.tensor (region r')).metadata.numel) :
    Memory.Apart ((b.view (region r)).address i) ((b.view (region r')).address j) := by
  have sep := b.admitted.separated r hr r' hr' different
  dsimp only at sep
  simp only [DataView.address,b.base _ (by omega : 0<(b.tensor (region r)).metadata.numel),
    b.base _ (by omega : 0<(b.tensor (region r')).metadata.numel),Memory.Apart]
  omega
theorem cells_apart (b : Buffers kind s f) (p : Program kind s) (a c : RegionName × Nat)
    (ha : a∈cells kind s f p) (hc : c∈cells kind s f p) (different : a≠c) :
    Memory.Apart ((b.view a.1).address a.2) ((b.view c.1).address c.2) := by
  have aa := (cells_coverage kind s f p a.1 a.2).mp ha
  have cc := (cells_coverage kind s f p c.1 c.2).mp hc
  by_cases same : a.1=c.1
  · have unequal : a.2≠c.2 := by intro eq; exact different (Prod.ext same eq)
    simp only [DataView.address,same,Memory.Apart]
    omega
  · obtain ⟨r,hr,ra,_⟩ := KernelGeometry.output_bounds kind s f p a.1 a.2 aa
    obtain ⟨r',hr',rc,_⟩ := KernelGeometry.output_bounds kind s f p c.1 c.2 cc
    have hd : r≠r' := by intro eq; subst r'; exact same (ra.symm.trans rc)
    rw [←ra,←rc]
    exact distinct_roles_apart b r r' (KernelGeometry.outputs_are_arguments kind f r hr)
      (KernelGeometry.outputs_are_arguments kind f r' hr') hd a.2 c.2
      (by rw [ra]; exact output_bound b p a.1 a.2 aa) (by rw [rc]; exact output_bound b p c.1 c.2 cc)

def placement (b : Buffers kind s f) (p : Program kind s) : Shared.Placement (KernelGeometry.Writes kind s f p) where
  tensor := b.tensor
  laws := b.laws
  checked := b.checked
  cells := cells kind s f p
  unique := List.nodup_dedup _
  coverage := cells_coverage kind s f p
  bounds := output_bound b p
  apart := fun a ha c hc different => cells_apart b p a c ha hc different

theorem native_output {Store Byte : Type} (api : Memory.API KernelNativeCarrier.Word Store Byte)
    (b : Buffers kind s f) (p : Program kind s) (isa : KernelNativeCarrier.Instructions)
    (routing : KernelNativeCarrier.Routing) (v : State KernelNativeCarrier.Node) (initial : Store)
    (r : RegionName) (i : Nat) (active : KernelGeometry.Writes kind s f p r i) :
    (logicalValue (fun a => (⟨api.read (Shared.commit api (placement b p)
      (fun _ _ => KernelNativeCarrier.observe isa routing) v initial) a⟩ : Word))
        (b.tensor r) (b.laws r) i).bits=KernelNativeCarrier.observe isa routing (v.mem r i) :=
  Shared.native_logical_output api (placement b p) (fun _ _ => KernelNativeCarrier.observe isa routing) v initial r i active

#axiomsClean physical_eq_logical
#axiomsClean native_input
#axiomsClean cells_coverage
#axiomsClean cells_apart
#axiomsClean native_output
#specNonCircular Buffers avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelTensor
