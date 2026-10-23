import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Contract
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Public
namespace VeriTile.Bench.Repairs.DequantizeMatmul.Direct
noncomputable section
set_option maxHeartbeats 600000

/-- Byte-width-indexed input view. Typed memory channels distinguish INT8 and
FP32 interpretations of storage; mixed-dtype raw aliases are rejected and the raw storage ID remains the
identity used by the actual output-alias guard. No materialization is made. -/
structure TensorView (bytes : Nat) where
  info : TensorInfo
  object : Nat
  offset : Nat
  physicalIndex : Nat → Nat

def TensorView.key (t : TensorView bytes) := 2*t.info.storage+(if bytes=1 then 1 else 0)
def TensorView.location (t : TensorView bytes) : Location := ⟨t.key,t.offset⟩
def TensorView.logicalAddress (t : TensorView bytes) (i : Nat) : Address := (t.key,t.physicalIndex i)

structure TensorLaws (storageBase : Nat → Nat) (t : TensorView bytes) : Prop where
  pointer : 0 < t.info.metadata.numel →
    t.info.metadata.dataPtr=BitVec.ofNat 64 (storageBase t.info.storage+bytes*t.offset)
  contiguous : t.info.metadata.isContiguous=Bool.true →
    ∀ i,i < t.info.metadata.numel → t.physicalIndex i=t.offset+i
  capacity : 0 < t.info.metadata.numel → storageBase t.info.storage+bytes*(t.offset+t.info.metadata.numel)≤2^64

structure Request where
  a : TensorView 4
  b : TensorView 1
  scale : TensorView 4
  out : Option (TensorView 4)
  contract : InputContract a.info b.info scale.info (out.map TensorView.info)

def requestShape (c : Request) := shape c.a.info c.b.info

def host (c : Request) (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key)
    (slive : caller c.scale.key) : HostInputs :=
  ⟨c.a.location,c.b.location,c.scale.location,c.a.info.metadata.device.index,caller,alive,blive,slive⟩

def descriptor (c : Request) (t : TensorView 4) : Descriptor :=
  ⟨t.object,t.location,rows c.a.info,columns c.b.info,t.info.metadata.device.index,
    canonical2 (rows c.a.info) (columns c.b.info)⟩
def provided (c : Request) : Option Descriptor := c.out.map (descriptor c)

theorem output_guard (c : Request) (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key)
    (slive : caller c.scale.key) : ∀ out,provided c=some out → OutGuard (requestShape c) (host c caller alive blive slive) out := by
  intro out h
  obtain ⟨t,ht,eq⟩:=Option.mem_map.mp h
  change c.out=some t at ht
  have hc:=c.contract.output t.info (by simp [ht])
  subst out
  refine ⟨rfl,rfl,congrArg VeriTile.Torch.Device.index hc.device,?_,?_,?_⟩
  · have hh:=hc.separated.1
    change 2*c.a.info.storage+(if 4=1 then 1 else 0) ≠ 2*t.info.storage+(if 4=1 then 1 else 0)
    simp only [show (4:Nat)≠1 by decide,↓reduceIte,Nat.add_zero]
    omega
  · change 2*c.b.info.storage+(if 1=1 then 1 else 0) ≠ 2*t.info.storage+(if 4=1 then 1 else 0)
    norm_num;omega
  · have hh:=hc.separated.2.2
    change 2*c.scale.info.storage+(if 4=1 then 1 else 0) ≠ 2*t.info.storage+(if 4=1 then 1 else 0)
    simp only [show (4:Nat)≠1 by decide,↓reduceIte,Nat.add_zero]
    omega

/-- The original typed storage, before allocation or source execution. -/
def logicalInput (c : Request) (zero : α) (m : Memory (Cell α)) : Input α (requestShape c) where
  a := fun r k=>(m (c.a.logicalAddress (r.val*columns c.a.info+k.val))).float zero
  b := fun k n=>(m (c.b.logicalAddress (k.val*columns c.b.info+n.val))).signed
  scale := fun n=>(m (c.scale.logicalAddress n.val)).float zero

/-- Ordinary dtype/byte-decoder laws for original live cells, with no result
or arithmetic premise. Same-dtype views may read aliased raw storage. -/
structure InputStorage (c : Request) (m : Memory (Cell α)) : Prop where
  a : ∀ i,i < c.a.info.metadata.numel → ∃ x,m (c.a.logicalAddress i)=.fp32 x
  b : ∀ i,i < c.b.info.metadata.numel → ∃ x,m (c.b.logicalAddress i)=.int8 x
  scale : ∀ i,i < c.scale.info.metadata.numel → ∃ x,m (c.scale.logicalAddress i)=.fp32 x

theorem flat_bound (r c row col : Nat) (hr : row < r) (hc : col < c) : row*c+col < r*c := by
  have hm:=Nat.mul_le_mul_right c hr
  simp only [Nat.succ_mul] at hm
  omega

theorem input_represents (c : Request) (storageBase : Nat → Nat)
    (la : TensorLaws storageBase c.a) (lb : TensorLaws storageBase c.b) (ls : TensorLaws storageBase c.scale)
    (zero : α) (m : Memory (Cell α)) (typed : InputStorage c m)
    (allocate : Allocator α) (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key) :
    Represents (logicalInput c zero m)
      (prepare allocate (requestShape c) (host c caller alive blive slive) (provided c)
        (output_guard c caller alive blive slive) m).buffers m := by
  constructor
  · intro r k
    have bound : r.val*columns c.a.info+k.val < c.a.info.metadata.numel := by
      rw [numel_matrix c.a.info c.contract.rank.1]
      exact flat_bound _ _ _ _ r.isLt k.isLt
    obtain ⟨x,hx⟩:=typed.a _ bound
    have idx:=la.contiguous c.contract.contiguous.1 _ bound
    change m (c.a.location.addr (r.val*columns c.a.info+k.val)) = .fp32 ((m (c.a.logicalAddress (r.val*columns c.a.info+k.val))).float zero)
    have address : c.a.location.addr (r.val*columns c.a.info+k.val)=c.a.logicalAddress (r.val*columns c.a.info+k.val) := by
      simp [TensorView.location,Location.addr,TensorView.logicalAddress,idx]
    rw [address,hx];rfl
  · intro k n
    have bound : k.val*columns c.b.info+n.val < c.b.info.metadata.numel := by
      rw [numel_matrix c.b.info c.contract.rank.2.1,←c.contract.shape.1]
      exact flat_bound _ _ _ _ k.isLt n.isLt
    obtain ⟨x,hx⟩:=typed.b _ bound
    have idx:=lb.contiguous c.contract.contiguous.2.1 _ bound
    change m (c.b.location.addr (k.val*columns c.b.info+n.val)) = .int8 ((m (c.b.logicalAddress (k.val*columns c.b.info+n.val))).signed)
    have address : c.b.location.addr (k.val*columns c.b.info+n.val)=c.b.logicalAddress (k.val*columns c.b.info+n.val) := by
      simp [TensorView.location,Location.addr,TensorView.logicalAddress,idx]
    rw [address,hx];rfl
  · intro n
    have bound : n.val < c.scale.info.metadata.numel := by
      simpa only [VeriTile.Torch.TensorMetadata.numel,c.contract.shape.2,List.prod_cons,List.prod_nil,Nat.mul_one] using (show n.val < columns c.b.info from n.isLt)
    obtain ⟨x,hx⟩:=typed.scale _ bound
    have idx:=ls.contiguous c.contract.contiguous.2.2 _ bound
    change m (c.scale.location.addr n.val) = .fp32 ((m (c.scale.logicalAddress n.val)).float zero)
    have address : c.scale.location.addr n.val=c.scale.logicalAddress n.val := by
      simp [TensorView.location,Location.addr,TensorView.logicalAddress,idx]
    rw [address,hx];rfl

def checkedRun (c : Request) (arithmetic : Arithmetic α) (allocate : Allocator α)
    (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key)
    (tile : Tile) (plan : LibraryPlan (requestShape c)) (before : Memory (Cell α)) : Except InputError (HostReturn α) := do
  checkTensors c.a.info c.b.info c.scale.info (c.out.map TensorView.info)
  return allocatedHostCall arithmetic allocate (requestShape c) (host c caller alive blive slive)
    (provided c) (output_guard c caller alive blive slive) tile plan before

theorem returned_device (c : Request) (arithmetic : Arithmetic α) (allocate : Allocator α)
    (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key)
    (tile : Tile) (plan : LibraryPlan (requestShape c)) (before : Memory (Cell α)) :
    (allocatedHostCall arithmetic allocate (requestShape c) (host c caller alive blive slive)
      (provided c) (output_guard c caller alive blive slive) tile plan before).tensor.device=c.a.info.metadata.device.index := by
  let chosen:=chooseAndAllocate allocate (requestShape c) (host c caller alive blive slive)
    (provided c) (output_guard c caller alive blive slive) before
  change (selectOutput (provided c) chosen.tensor).device = _
  rw [chosen.chosen]
  exact chosen.guard.device

#axiomsClean input_represents
#axiomsClean returned_device
end
end VeriTile.Bench.Repairs.DequantizeMatmul.Direct
