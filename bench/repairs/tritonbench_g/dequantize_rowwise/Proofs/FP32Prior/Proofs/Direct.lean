import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Contract
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Native
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Direct
open VeriTile.Triton FP.Structural Bytes
noncomputable section
abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev TensorLaws := VeriTile.Torch.FP32.TensorLaws
structure Request where
  x : Tensor
  scale : Tensor
  xLaws : TensorLaws x
  scaleLaws : TensorLaws scale
  contract : InputContract x.metadata scale.metadata

def xView (c : Request) := VeriTile.Torch.FP32.dataView c.x c.xLaws c.contract.contiguous.1
def scaleView (c : Request) := VeriTile.Torch.FP32.dataView c.scale c.scaleLaws c.contract.contiguous.2
def p (c : Request) := parameters c.x.metadata
abbrev Coordinates (c : Request) := Coord (rows c.x.metadata) (columns c.x.metadata)

def logical (memory : Bytes.Memory) (t : Tensor) (i : Nat) : Word :=
  let w := Memory.read32 memory (t.storage.address (t.physicalIndex i))
  if t.metadata.isNeg then (fun b=>if b.val=3 then w b ^^^ 128 else w b) else w

theorem direct_logical (memory : Bytes.Memory) (t : Tensor) (laws : TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (noNeg : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    Memory.read32 memory ((VeriTile.Torch.FP32.dataView t laws contiguous).base+4*i)=logical memory t i := by
  simp only [logical,noNeg,Bool.false_eq_true,↓reduceIte,laws.contiguous contiguous i hi]
  congr 1
  simp only [VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,
    if_neg (by omega : t.metadata.numel≠0),VeriTile.Torch.FP32.Storage.address]
  omega

theorem nonempty_coordinate (c : Request) (q : Coordinates c) : 0<c.x.metadata.numel := by
  rw [numel_eq _ _ c.contract]
  exact Nat.mul_pos (Nat.zero_lt_of_lt q.1.isLt) (Nat.zero_lt_of_lt q.2.isLt)

structure Allocation (c : Request) where
  object : Nat
  metadata : TensorMetadata
  base : Nat
  pointer : metadata.dataPtr=BitVec.ofNat 64 base
  shape : metadata.shape=c.x.metadata.shape
  dtype : metadata.dtype=.other "float16"
  device : metadata.device=c.x.metadata.device
  contiguous : metadata.isContiguous=Bool.true
  noNeg : metadata.isNeg=Bool.false
  objects : object≠c.x.object ∧ object≠c.scale.object
  spanFits : base+2*c.x.metadata.numel≤2^64
  alignment : base%2=0
  separated : @Memory.Separated (rows c.x.metadata) (columns c.x.metadata)
    (xView c).base (scaleView c).base base

def bounds (c : Request) (out : Allocation c) (nonempty : 0<c.x.metadata.numel) :
    Native.Bounds (p c) (xView c).base (scaleView c).base out.base c.scale.metadata.numel := by
  refine ⟨c.contract.grid,padded_bound _ _ c.contract,c.contract.scales nonempty,?_,?_,?_,?_⟩
  · have small:=numel_bound _ _ c.contract
    rw [numel_eq _ _ c.contract] at small
    change 4*(rows c.x.metadata*columns c.x.metadata)≤2^63-1
    omega
  · have span := (xView c).spanFits
    change (xView c).base+4*c.x.metadata.numel≤2^64 at span
    simpa [p,parameters,numel_eq _ _ c.contract] using span
  · exact (scaleView c).spanFits
  · simpa [p,parameters,numel_eq _ _ c.contract] using out.spanFits

theorem x_coordinate (c : Request) (m : Bytes.Memory) (q : Coordinates c) :
    Memory.read32 m ((xView c).base+4*linear q)=logical m c.x (linear q) := by
  apply direct_logical m c.x c.xLaws c.contract.contiguous.1 c.contract.noNeg.1
  simpa [numel_eq _ _ c.contract] using linear_bound q

theorem scale_coordinate (c : Request) (m : Bytes.Memory) (q : Coordinates c) :
    Memory.read32 m ((scaleView c).base+4*q.1.val)=logical m c.scale q.1.val := by
  apply direct_logical m c.scale c.scaleLaws c.contract.contiguous.2 c.contract.noNeg.2
  exact lt_of_lt_of_le q.1.isLt (c.contract.scales (nonempty_coordinate c q))

theorem actual_pointers (c : Request) (nonempty : 0<c.x.metadata.numel) :
    BitVec.ofNat 64 (xView c).base=c.x.dataPtr ∧
    BitVec.ofNat 64 (scaleView c).base=c.scale.dataPtr := by
  constructor
  · exact VeriTile.Torch.FP32.actual_pointer c.x c.xLaws c.contract.contiguous.1 nonempty
  · apply VeriTile.Torch.FP32.actual_pointer c.scale c.scaleLaws c.contract.contiguous.2
    have hr : 0<rows c.x.metadata := by
      rw [numel_eq _ _ c.contract] at nonempty
      by_contra bad
      have zero : rows c.x.metadata=0 := by omega
      simp [zero] at nonempty
    exact lt_of_lt_of_le hr (c.contract.scales nonempty)

structure Returned where
  object : Nat
  metadata : TensorMetadata
  memory : Bytes.Memory

def run (c : Request) (out : Allocation c) (ops : Memory.Numerics) (m : Bytes.Memory) : Except InputError Returned := do
  checkTensorInputs c.x.metadata c.scale.metadata
  return ⟨out.object,out.metadata,if c.x.metadata.numel=0 then m
    else Native.launch ops (p c) (xView c).base (scaleView c).base out.base m⟩

def Result (c : Request) (out : Allocation c) (ops : Memory.Numerics)
    (before : Bytes.Memory) (ret : Returned) : Prop :=
  ret.object=out.object ∧ ret.metadata=out.metadata ∧
  ret.metadata.shape=c.x.metadata.shape ∧ ret.metadata.dtype=.other "float16" ∧
  ret.metadata.device=c.x.metadata.device ∧
  (∀ q : Coordinates c,Memory.read16 ret.memory (out.base+2*linear q)=
    Memory.compute ops (logical before c.scale q.1.val) (logical before c.x (linear q))) ∧
  (∀ byte,(∀ e : Memory.Event (rows c.x.metadata) (columns c.x.metadata),byte≠Memory.key out.base e) →
    ret.memory byte=before byte)

theorem run_correct (c : Request) (out : Allocation c) (ops : Memory.Numerics) (m : Bytes.Memory) :
    ∃ ret,run c out ops m=.ok ret ∧ Result c out ops m ret := by
  let after := if c.x.metadata.numel=0 then m else Native.launch ops (p c) (xView c).base (scaleView c).base out.base m
  refine ⟨⟨out.object,out.metadata,after⟩,?_,rfl,rfl,out.shape,out.dtype,out.device,?_,?_⟩
  · simp [run,(checkTensorInputs_iff _ _).mpr c.contract,after]
  · intro q
    have positive:=nonempty_coordinate c q
    change Memory.read16 after _=_
    simp only [after,if_neg (Nat.ne_of_gt positive)]
    apply (Native.launch_readback ops (p c) (bounds c out positive) out.separated m q).trans
    change Memory.compute ops (Memory.read32 m ((scaleView c).base+4*q.1.val))
      (Memory.read32 m ((xView c).base+4*linear q))=_
    rw [x_coordinate,scale_coordinate]
  · intro byte outside
    change after byte=m byte
    by_cases empty : c.x.metadata.numel=0
    · simp [after,empty]
    · simp only [after,if_neg empty]
      exact Native.launch_frame ops (p c) (bounds c out (Nat.pos_of_ne_zero empty)) out.separated m byte outside

/-- Initial typed input regions read directly from the entry byte heap. -/
def initial (c : Request) (out : Allocation c) (m : Bytes.Memory) : State Word where
  mem := fun region index=>if region=xRegion then .mk .real (Memory.read32 m ((xView c).base+4*index))
    else if region=scaleRegion then .mk .real (Memory.read32 m ((scaleView c).base+4*index))
    else .mk .fp16 (Memory.pad (Memory.read16 m (out.base+2*index)))
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun dim=>if dim=0 then rows c.x.metadata else 1
  undef := fun d _ _=>defaultValue d

def sourceStore (c : Request) (out : Allocation c) (m : Bytes.Memory) (t : State Word) : Bytes.Memory :=
  Scatter.run (Memory.events (rows c.x.metadata) (columns c.x.metadata)) (Memory.key out.base)
    (fun e=>Memory.low ((t.mem outRegion (linear e.1)).read .fp16) e.2) m

theorem typed_native (rest : Algebra Word) (c : Request) (out : Allocation c)
    (ops : Memory.Numerics) (m : Bytes.Memory) (positive : 0<c.x.metadata.numel) :
    ∃ t,Grid.launch (Memory.backend rest ops) (p c) (initial c out m)=some t ∧
      Native.launch ops (p c) (xView c).base (scaleView c).base out.base m=sourceStore c out m t := by
  obtain ⟨t,ht,read,frame⟩:=Grid.launch_correct (Memory.backend rest ops) (p c) (initial c out m)
  have readback : ∀ q : Coordinates c,Memory.pad
      (Memory.read16 (Native.launch ops (p c) (xView c).base (scaleView c).base out.base m)
        (out.base+2*linear q))=(t.mem outRegion (linear q)).read .fp16 := by
    intro q
    apply (congrArg Memory.pad (Native.launch_readback ops (p c) (bounds c out positive) out.separated m q)).trans
    have hr:=read q.1.val q.1.isLt q.2.val q.2.isLt
    rw [show linear q=q.1.val*(p c).cols+q.2.val from rfl,hr]
    simp only [Cell.read_same,Grid.value,initial,xRegion,scaleRegion,Region.mk.injEq,
      String.reduceEq,↓reduceIte,Memory.typed_scalar,Memory.value,Memory.scaleAddress,Memory.xAddress]
    rfl
  refine ⟨t,ht,?_⟩
  funext byte
  classical
  by_cases hit : ∃ e : Memory.Event (rows c.x.metadata) (columns c.x.metadata),byte=Memory.key out.base e
  · obtain ⟨e,rfl⟩:=hit
    have readByte:=congrArg (fun w=>Memory.low w e.2) (readback e.1)
    simp only [Memory.low_pad] at readByte
    change Memory.read16 (Native.launch ops (p c) (xView c).base (scaleView c).base out.base m)
      (out.base+2*linear e.1) e.2=_
    rw [readByte]
    symm
    exact Scatter.readback_injective _ _ _ _ e (Memory.events_complete e) (Memory.key_injective out.base)
  · have miss : ∀ e : Memory.Event (rows c.x.metadata) (columns c.x.metadata),byte≠Memory.key out.base e := by
      intro e equal;exact hit ⟨e,equal⟩
    rw [Native.launch_frame ops (p c) (bounds c out positive) out.separated m byte miss]
    symm
    exact Scatter.frame _ _ _ _ byte (fun e _=>miss e)

#axiomsClean run_correct
#axiomsClean typed_native
#axiomsClean actual_pointers
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Direct
