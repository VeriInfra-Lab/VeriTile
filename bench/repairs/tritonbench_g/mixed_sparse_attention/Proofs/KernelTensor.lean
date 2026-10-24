import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelGrid
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelContract
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Reify
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelStorage

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelTensor
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid
open VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Shared
export VeriTile.Bench.Repairs.FastLayernorm.KernelStorage
  (Placement TensorImage commit complete commit_output commit_frame native_logical_output)
end Shared
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Ordinary Tensor object, metadata and word-storage observations. Integer
metadata uses int32 laws; it is not assigned contradictory FP32 TensorLaws. -/
structure Observation (raw : Checked.Tensor) (t : Torch.FP32.Tensor) : Prop where
  object : t.object=raw.object
  shape : t.metadata.shape=raw.shape
  dtype : t.metadata.dtype=if raw.floating then .floating .fp32 else .other "int32"
  strided : t.metadata.isStrided=raw.strided
  cuda : t.metadata.isCuda=raw.cuda
  device : t.metadata.device.index=raw.device
  contiguous : t.metadata.isContiguous=raw.contiguous
  negative : t.metadata.isNeg=raw.negative
  pointer : t.dataPtr.toNat=raw.pointer
  storageBase : t.storage.base=raw.storageBase
  storageBytes : 4*t.storage.elements=raw.storageBytes
  storageOffset : t.storageOffset=raw.storageOffset
  bounds : ∀ i,i<t.metadata.numel → t.physicalIndex i<t.storage.elements
  first : 0<t.metadata.numel → t.dataPtr=BitVec.ofNat 64 (t.storage.address t.storageOffset)
  index : t.metadata.isContiguous=Bool.true → ∀ i,i<t.metadata.numel → t.physicalIndex i=t.storageOffset+i
structure Buffers (request : Request) where
  tensor : Role → Torch.FP32.Tensor
  observation : ∀ r,Observation (request.tensor r) (tensor r)
  strides : Role → List Nat
  conjugate : Role → Bool
  strideObservation : ∀ r,strides r=(request.tensor r).strides
  conjugateObservation : ∀ r,conjugate r=(request.tensor r).conjugate

theorem count (b : Buffers request) (accepted : Admission c request values) (r : Role) :
    (b.tensor r).metadata.numel=r.size c := by
  rw [Torch.TensorMetadata.numel,(b.observation r).shape,(accepted.schema.header r).1]
  cases r <;> simp [Role.shape,Role.size,tensorSize,Nat.mul_assoc]
theorem logical_address (b : Buffers request) (accepted : Admission c request values)
    (r : Role) (i : Nat) (hi : i<r.size c) :
    (b.tensor r).storage.address ((b.tensor r).physicalIndex i)=(request.tensor r).pointer+4*i := by
  have dense:=(accepted.schema.header r).2.2.2.2.1
  rw [(b.observation r).index ((b.observation r).contiguous.trans dense) i (by rw [count b accepted];exact hi)]
  simp only [Torch.FP32.Storage.address,(b.observation r).storageBase,(b.observation r).storageOffset]
  rw [(accepted.storage r).1]
  omega
def fpLaws (b : Buffers request) (accepted : Admission c request values)
    (r : Role) (hr : r.floating=Bool.true) : Torch.FP32.TensorLaws (b.tensor r) where
  dtype:=by rw [(b.observation r).dtype,(accepted.schema.header r).2.1,hr];rfl
  layout:=by
    have h:=(b.observation r).strided.trans (accepted.schema.header r).2.2.2.2.2.2.2
    simpa [Torch.TensorMetadata.isStrided] using h
  inBounds:=(b.observation r).bounds
  pointer:=(b.observation r).first
  contiguous:=(b.observation r).index
def outputChecked (b : Buffers request) (accepted : Admission c request values) :
    VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract (b.tensor .output).metadata Bool.true where
  layout:=(fpLaws b accepted .output rfl).layout
  dtype:=(fpLaws b accepted .output rfl).dtype
  cuda:=by
    have h:=(b.observation .output).cuda.trans (accepted.schema.header .output).2.2.1
    simpa [Torch.TensorMetadata.isCuda] using h
  dense:=fun _=>(b.observation .output).contiguous.trans (accepted.schema.header .output).2.2.2.2.1
  noNeg:=(b.observation .output).negative.trans (accepted.schema.header .output).2.2.2.2.2.1
  alignment:=fun _=>by rw [show (b.tensor .output).metadata.dataPtr=(b.tensor .output).dataPtr from rfl,
    (b.observation .output).pointer];exact (accepted.pointers .output).1

def readRole (c : Config) (request : Request) (read : Nat → β) (zero : β) (r : Role) (i : Nat) : β :=
  if i<r.size c then read ((request.tensor r).pointer+4*i) else zero
def observed (c : Config) (request : Request) (read : Nat → β) (zero : β) (integer : β → Int) : Observed :=
  fun r i=>integer (readRole c request read zero r i)
def initial [Inhabited α] (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → β) (zero : β) (encode : β → α) (integer : β → Int) : State α where
  mem:=fun r i=>if r=c.out then .mk .real (encode (readRole c request read zero .output i))
    else match layout.route r with
      | some role=>if role.floating then .mk .real (encode (readRole c request read zero role i))
        else .mk .nat (integer (readRole c request read zero role i)).toNat
      | none=>.mk .real (encode zero)
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>1
  undef:=fun dtype _ _=>defaultValue dtype
theorem input_apart (layout : Reify.Layout c) (r : Role) (hr : r∈inputRoles) : r.region c≠c.out := by
  intro h
  have eq:=layout.inputs r hr
  rw [h,layout.output] at eq
  contradiction
theorem input_cell [Inhabited α] (layout : Reify.Layout c) (r : Role) (hr : r∈inputRoles)
    (read : Nat → β) (zero : β) (encode : β → α) (integer : β → Int) (i : Nat) :
    (initial c request layout read zero encode integer).mem (r.region c) i=
      if r.floating then .mk .real (encode (readRole c request read zero r i))
      else .mk .nat (integer (readRole c request read zero r i)).toNat := by
  simp [initial,input_apart layout r hr,layout.inputs r hr]
theorem metadata_initial [Inhabited α] (layout : Reify.Layout c) (read : Nat → β)
    (zero : β) (encode : β → α) (integer : β → Int) :
    MetadataBinding c (observed c request read zero integer) (initial c request layout read zero encode integer) := by
  constructor
  intro r hr i
  have member : r∈inputRoles := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl|rfl <;> simp [inputRoles]
  rw [input_cell layout r member]
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl|rfl|rfl <;> rfl

def cells [Inhabited α] (c : Config) (s : State α) : List (RegionName × Nat) :=
  ((Raw.all c).filter (fun q=>decide (live c s q)) |>.map (fun q=>(c.out,Raw.offset q))).dedup
theorem cells_coverage [Inhabited α] (c : Config) (s : State α) (r : RegionName) (i : Nat) :
    (r,i)∈cells c s ↔ Writes c s r i := by
  simp only [cells,List.mem_dedup,List.mem_map,List.mem_filter,Raw.coverage,true_and,decide_eq_true_eq,
    Prod.mk.injEq,Writes]
  constructor
  · rintro ⟨q,hq,hr,hi⟩
    exact ⟨hr.symm,q,hq,hi.symm⟩
  · rintro ⟨hr,q,hq,hi⟩
    exact ⟨q,hq,hr.symm,hi.symm⟩
def placement [Inhabited α] (b : Buffers request) (accepted : Admission c request values) (s : State α) :
    Shared.Placement (Writes c s) where
  tensor:=fun _=>b.tensor .output
  laws:=fun _=>fpLaws b accepted .output rfl
  checked:=fun _=>outputChecked b accepted
  cells:=cells c s
  unique:=List.nodup_dedup _
  coverage:=cells_coverage c s
  bounds:=by
    rintro r i ⟨_,q,_,rfl⟩
    rw [count b accepted]
    exact Raw.offset_bound q
  apart:=by
    intro a ha z hz ne
    have aa:=(cells_coverage c s a.1 a.2).mp ha
    have zz:=(cells_coverage c s z.1 z.2).mp hz
    have hn : a.2≠z.2 := by intro eq;exact ne (Prod.ext (aa.1.trans zz.1.symm) eq)
    simp only [Torch.FP32.DataView.address,Memory.Apart]
    omega

theorem placement_address [Inhabited α] (b : Buffers request) (accepted : Admission c request values)
    (s : State α) (r : RegionName) (i : Nat) (hi : Writes c s r i) :
    (placement b accepted s).address r i=(request.tensor .output).pointer+4*i := by
  rw [←VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.logical_address _ r i hi]
  exact logical_address b accepted .output i (by obtain ⟨_,q,_,rfl⟩:=hi;exact Raw.offset_bound q)

#axiomsClean logical_address
#axiomsClean metadata_initial
#axiomsClean cells_coverage
#axiomsClean placement_address
#specNonCircular Buffers avoiding [runPrograms,FP.Structural.exec,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelTensor
