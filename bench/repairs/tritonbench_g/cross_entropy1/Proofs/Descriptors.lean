import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Storage

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

structure Permissions (σ : Type) where
  readable : σ → Nat → Nat → Prop
  writable : σ → Nat → Nat → Prop

/-- Actual tensor-object metadata and byte permissions. Backing is stated
only for logical coordinates of this descriptor. It imposes no artificial
class-count, stride, offset, or rounded-tail allocation bound. -/
structure Descriptor {σ : Type} (layout : Layout σ) (permissions : Permissions σ)
    (native : σ) (t : Tensor) (width : Nat) : Prop where
  active : layout.active native t.region
  device : (layout.buffer native t.region).device=t.device
  backing : ∀ i, i<t.numel → width*(t.index i+1)≤(layout.buffer native t.region).capacity
  readable : ∀ i j, i<t.numel → j<width →
    permissions.readable native t.device ((layout.buffer native t.region).base+width*t.index i+j)

structure WritableDescriptor {σ : Type} (layout : Layout σ) (permissions : Permissions σ)
    (native : σ) (t : Tensor) (width : Nat) : Prop extends Descriptor layout permissions native t width where
  writable : ∀ i j, i<t.numel → j<width →
    permissions.writable native t.device ((layout.buffer native t.region).base+width*t.index i+j)

def dataPointer {σ : Type} (layout : Layout σ) (native : σ) (t : Tensor) (width : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (layout.buffer native t.region).base + BitVec.ofNat 64 t.storageOffset * BitVec.ofNat 64 width

theorem view_pointer_word {σ : Type} (layout : Layout σ) (native : σ) (t : Tensor) (width delta : Nat) :
    dataPointer layout native t width + BitVec.ofNat 64 delta * BitVec.ofNat 64 width=
      Native.address (layout.buffer native t.region).base width (BitVec.ofNat 64 (t.storageOffset+delta)) := by
  simp [dataPointer,Native.address,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.add_mul,Nat.add_assoc]

theorem descriptor_address {σ : Type} (layout : Layout σ) (permissions : Permissions σ)
    (native : σ) (t : Tensor) (width : Nat) (descriptor : Descriptor layout permissions native t width)
    (positive : 0<width) (i : Nat) (inside : i<t.numel) :
    (Native.address (layout.buffer native t.region).base width (BitVec.ofNat 64 (t.index i))).toNat=
        (layout.buffer native t.region).base+width*t.index i ∧
    (layout.buffer native t.region).base+width*t.index i+width≤2^64 ∧
    ∀ j, j<width → permissions.readable native t.device
      ((Native.address (layout.buffer native t.region).base width (BitVec.ofNat 64 (t.index i))).toNat+j) := by
  have h := Native.backed_address (layout.buffer native t.region) (t.index i) width positive
    (descriptor.backing i inside) (layout.addressSpace native t.region descriptor.active)
  exact ⟨h.1,h.2.2,by intro j hj; rw [h.1]; exact descriptor.readable i j inside hj⟩

theorem row_coordinate_bound (t : Tensor) (rows cols row col : Nat) (shape : t.shape=[rows,cols])
    (hr : row<rows) (hc : col<cols) : row*cols+col<t.numel := by
  simp only [Tensor.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
  have product := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at product
  omega

/-- The actual tensor geometry discharges the capacity and signed-index
bounds for a strided/offset FP32 source or destination. -/
theorem strided_address {σ : Type} (layout : Layout σ) (permissions : Permissions σ)
    (native : σ) (t : Tensor) (descriptor : Descriptor layout permissions native t 4)
    (rows cols width count row block lane : Nat) (shape : t.shape=[rows,cols]) (geometry : RowLayout t)
    (api : Native.Supported rows count width cols) (hr : row<rows) (hb : block<count) (hl : lane<width)
    (active : block*width+lane<cols) :
    (Native.address (layout.buffer native t.region).base 4
      (Native.element t.storageOffset row (rowStride t) width block lane)).toNat=
        (layout.buffer native t.region).base+4*(t.storageOffset+row*rowStride t+(block*width+lane)) ∧
    (BitVec.ofNat 64 (t.storageOffset+row*rowStride t+(block*width+lane))).toInt=
      (t.storageOffset+row*rowStride t+(block*width+lane) : Nat) ∧
    ∀ byte, byte<4 → permissions.readable native t.device
      ((layout.buffer native t.region).base+4*(t.storageOffset+row*rowStride t+(block*width+lane))+byte) := by
  have bound := row_coordinate_bound t rows cols row (block*width+lane) shape hr active
  have coordinate := geometry rows cols shape row (block*width+lane) hr active
  have backed : 4*(t.storageOffset+row*rowStride t+(block*width+lane)+1)≤(layout.buffer native t.region).capacity := by
    simpa only [coordinate] using descriptor.backing _ bound
  have address := descriptor_address layout permissions native t 4 descriptor (by decide) _ bound
  have signed := Native.fp32_index_signed (layout.buffer native t.region)
    (t.storageOffset+row*rowStride t+(block*width+lane)) backed (layout.addressSpace native t.region descriptor.active)
  refine ⟨?_,signed,?_⟩
  · rw [Native.element_word t.storageOffset row (rowStride t) width block lane rows count cols api hr hb hl]
    simpa only [coordinate] using address.1
  · intro byte hbyte
    simpa only [coordinate] using descriptor.readable (row*cols+(block*width+lane)) byte bound hbyte

/-- Masked lanes need no allocation backing. Their range predicate is exact;
only an active coordinate invokes the descriptor-address theorem. -/
theorem strided_mask {σ : Type} (layout : Layout σ) (permissions : Permissions σ)
    (native : σ) (t : Tensor) (descriptor : Descriptor layout permissions native t 4)
    (rows cols width count row block lane : Nat) (shape : t.shape=[rows,cols]) (geometry : RowLayout t)
    (api : Native.Supported rows count width cols) (hr : row<rows) (hb : block<count) (hl : lane<width) :
    Native.columnMask cols width block lane=decide (block*width+lane<cols) ∧
    (Native.columnMask cols width block lane=Bool.true →
      (Native.address (layout.buffer native t.region).base 4
        (Native.element t.storageOffset row (rowStride t) width block lane)).toNat+4≤2^64) := by
  have mask := (Native.source_mask rows count width cols block lane api hb hl).2
  refine ⟨mask,?_⟩
  intro selected
  have active : block*width+lane<cols := by simpa only [mask,decide_eq_true_eq] using selected
  have bound := row_coordinate_bound t rows cols row (block*width+lane) shape hr active
  have coordinate := geometry rows cols shape row (block*width+lane) hr active
  have address := descriptor_address layout permissions native t 4 descriptor (by decide) _ bound
  rw [(strided_address layout permissions native t descriptor rows cols width count row block lane shape geometry api hr hb hl active).1]
  simpa only [coordinate] using address.2.1

#axiomsClean view_pointer_word
#axiomsClean descriptor_address
#axiomsClean strided_address
#axiomsClean strided_mask
#specNonCircular Descriptor avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#specNonCircular WritableDescriptor avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
