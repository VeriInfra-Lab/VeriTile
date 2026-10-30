import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Tensor
open VeriTile.Triton FP.Structural
open MatmulDequantInt4.Direct
noncomputable section
set_option maxHeartbeats 4000000

def dtype : MatmulDequantInt4.Host.DType → VeriTile.Torch.DType
  | .fp16 => .other "torch.float16"
  | .fp32 => .floating .fp32
  | .int32 => .other "torch.int32"

/-- Actual Tensor storage and coordinate observations, with shared Torch
metadata. These laws describe only the input ABI, never execution or results. -/
structure Actual (info : Info) where
  metadata : VeriTile.Torch.TensorMetadata
  storageObject : Nat
  physicalIndex : Nat → Nat → Nat
  dtype_eq : metadata.dtype=dtype info.tensor.dtype
  layout_eq : metadata.layout=.strided
  device_eq : metadata.device=⟨"cuda",info.tensor.device⟩
  shape_eq : metadata.shape=[info.tensor.rows,info.tensor.columns]
  contiguous_eq : metadata.isContiguous=info.contiguous
  neg_eq : metadata.isNeg=info.tensor.lazyNeg
  pointer_eq : metadata.dataPtr=BitVec.ofNat 64 info.tensor.base
  coordinates : ∀ r n,physicalIndex r n=info.tensor.offset+info.tensor.strides.address r n

structure Arguments (p : PackedInfo) (o : Info) where
  weight : Actual p.weight
  scale : Actual p.scale
  zero : Actual p.zero
  output : Actual o
  output_storage : ∀ t∈[p.weight.tensor,p.scale.tensor,p.zero.tensor],
    ∀ storage,((t=p.weight.tensor ∧ storage=weight.storageObject) ∨
      (t=p.scale.tensor ∧ storage=scale.storageObject) ∨
      (t=p.zero.tensor ∧ storage=zero.storageObject)) →
    storage=output.storageObject → t.base=o.tensor.base ∧ t.storageBytes=o.tensor.storageBytes

/-- Typed allocation observations retain all 16/32-bit payloads and INT32
packed words without imposing a four-byte capacity on FP16 tensors. -/
structure Heap (α : Type) where
  values : Nat → Nat → α
  packed : Nat → Nat → BitVec 32

/-- Bind the same Tensor storage at each actual data-pointer origin. -/
def entry {α : Type} [Inhabited α] (f : Source.Profile) (p : PackedInfo) (o : Info)
    (t : Arguments p o) (heap : Heap α) : State α :=
  { regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 0
    undef := fun d _ _ => defaultValue d
    mem := fun r a =>
    if r="Q" then .mk .int (heap.packed t.weight.storageObject (p.weight.tensor.offset+a)).toInt
    else if r="Z" then .mk .int (heap.packed t.zero.storageObject (p.zero.tensor.offset+a)).toInt
    else if r="S" then Source.cell f (heap.values t.scale.storageObject (p.scale.tensor.offset+a))
    else Source.cell f (heap.values t.output.storageObject (o.tensor.offset+a)) }

def read {α : Type} [Inhabited α] (f : Source.Profile) (c : Cell α) : α :=
  match f with | .half => c.read .fp16 | .single => c.read .fp32

@[simp] theorem read_cell {α : Type} [Inhabited α] (f : Source.Profile) (a : α) :
    read f (Source.cell f a)=a := by cases f <;> rfl

/-- Address renaming of the source stores to the existing destination Tensor.
No allocator, returned-object or host execution is assumed. -/
def physical {α : Type} [Inhabited α] (f : Source.Profile) (p : PackedInfo) (o : Info)
    (t : Arguments p o) (heap : Heap α) (state : State α) : Heap α :=
  {heap with values := fun r a =>
    if r=t.output.storageObject ∧ o.tensor.offset≤a then read f (state.mem "O" (a-o.tensor.offset))
    else heap.values r a}

def Outside (p : PackedInfo) (o : Info) (t : Arguments p o) (r a : Nat) : Prop :=
  r≠t.output.storageObject ∨ a<o.tensor.offset ∨
    ∀ k n,k<p.weight.tensor.rows*8 → n<p.weight.tensor.columns →
      a≠t.output.physicalIndex k n

def Post {α : Type} (p : PackedInfo) (o : Info) (t : Arguments p o)
    (target : Nat → Nat → α) (initial final : Heap α) : Prop :=
  (∀ k n,k<p.weight.tensor.rows*8 → n<p.weight.tensor.columns →
    final.values t.output.storageObject (t.output.physicalIndex k n)=target k n) ∧
  (∀ r a,Outside p o t r a → final.values r a=initial.values r a) ∧
  final.packed=initial.packed

theorem whole {α : Type} [Inhabited α] (f : Source.Profile) (M : Algebra α)
    (p : PackedInfo) (o : Info) (t : Arguments p o)
    (h : PackedContract p 1 f.hostDtype) (ho : OutputABI p f o) (i : Fin 8) (heap : Heap α) :
    ∃ final,KernelGrid.run f M (params p i) (entry f p o t heap)=some final ∧
      Post p o t (fun k n =>
        if hk : k<p.weight.tensor.rows*8 then if hn : n<p.weight.tensor.columns then
          KernelGrid.result f M (params p i) (entry f p o t heap) (⟨k,hk⟩,⟨n,hn⟩)
        else default else default) heap (physical f p o t heap final) := by
  obtain ⟨state,executed,values,frame⟩ := KernelGrid.whole f M (params p i) (valid p i) (entry f p o t heap)
  have coordinates (k n) : t.output.physicalIndex k n=o.tensor.offset+(k*p.weight.tensor.columns+n) := by
    rw [t.output.coordinates,ho.2.1]
    simp [MatmulDequantInt4.Strides.canonical,MatmulDequantInt4.Strides.Layout.address]
  refine ⟨state,executed,?_,?_,rfl⟩
  · intro k n hk hn
    have hv := congrArg (read f) (values (⟨k,hk⟩,⟨n,hn⟩) trivial)
    simp only [read_cell] at hv
    rw [coordinates]
    simpa [physical,KernelGrid.address,params,hk,hn] using hv
  · intro r a outside
    by_cases hw : r=t.output.storageObject ∧ o.tensor.offset≤a
    · have exterior : ∀ x : KernelGrid.Coordinate (params p i),
          a-o.tensor.offset≠KernelGrid.address (params p i) x := by
        intro x eq
        rcases outside with hr|ha|hc
        · exact hr hw.1
        · omega
        · apply hc x.1.val x.2.val x.1.isLt x.2.isLt
          rw [coordinates]
          change a=o.tensor.offset+KernelGrid.address (params p i) x
          omega
      have hf := congrArg (read f) (frame "O" (a-o.tensor.offset) (Or.inr (fun x _ => exterior x)))
      simp only [entry,show ("O":RegionName)≠"Q" by decide,
        show ("O":RegionName)≠"Z" by decide,show ("O":RegionName)≠"S" by decide,
        ite_false,read_cell,Nat.add_sub_of_le hw.2] at hf
      simpa only [physical,hw.1,hw.2,and_self,ite_true] using hf
    · simp [physical,hw]

#specNonCircular Post avoiding [KernelGrid.run,Source.source,Source.execution,physical]
#axiomsClean whole
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Tensor
