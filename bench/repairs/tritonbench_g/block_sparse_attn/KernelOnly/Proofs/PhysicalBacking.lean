import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PhysicalInputs
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.InputSafety
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
variable {α β Store Byte : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

def capacity (storage : BlockSparseAttn.Safety.Storage) : Nat := storage.bytes/4-storage.offset

theorem allocated_bound (storage : BlockSparseAttn.Safety.Storage) (address : Int)
    (allocated : BlockSparseAttn.Safety.Allocated storage address) : address.toNat<capacity storage := by
  have backed : storage.offset+address.toNat+1≤storage.bytes/4 :=
    (Nat.le_div_iff_mul_le (by decide)).mpr (by simpa only [Nat.mul_comm] using allocated.2)
  unfold capacity
  omega

def bufferBounds (buffers : Physical.Buffers) (region : RegionName) : Nat :=
  if region="rows" then capacity (buffers.input .rows).storage
  else if region="cols" then capacity (buffers.input .cols).storage
  else if region="Q" then capacity (buffers.input .query).storage
  else if region="K" then capacity (buffers.input .key).storage
  else if region="V" then capacity (buffers.input .value).storage
  else if region="out" then capacity buffers.output.storage else 0

def physicalLayout (access : PhysicalKernel.Access β Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (m : Store) : BlockSparseAttn.Layout :=
  layoutOf p batches (PhysicalKernel.metadata access (buffers.input .rows) m)
    (PhysicalKernel.metadata access (buffers.input .cols) m)

theorem row_div (r : Nat) (i : Fin 16) : (r*16+i.val)/16=r := by
  omega

theorem snapshot_count (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
    (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store)
    (r : Row p) (h : Head p batches) (i : Fin 16) :
    (physicalLayout access p batches buffers m).csr.count (physicalLayout access p batches buffers m).shape
        (head p h.val) (r.val*16+i.val)=
      count (snapshot access zero decode p batches visits buffers m) r h := by
  change (PhysicalKernel.integer access (buffers.input .rows) m
      (layout p h.val*p.rowStride+(r.val*16+i.val)/16+1)-
    PhysicalKernel.integer access (buffers.input .rows) m
      (layout p h.val*p.rowStride+(r.val*16+i.val)/16)).toNat=
    (PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val+1)-
    PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val+0)).toNat
  rw [row_div]
  rfl

theorem snapshot_key (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
    (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store)
    (r : Row p) (h : Head p batches) (i : Fin 16) (v : Fin visits) (n : Nat) :
    (physicalLayout access p batches buffers m).csr.key (physicalLayout access p batches buffers m).shape
        (head p h.val) (r.val*16+i.val) v.val n=
      column (snapshot access zero decode p batches visits buffers m) r h v*16+n := by
  change PhysicalKernel.integer access (buffers.input .cols) m
      ((layout p h.val*p.colStride : Nat)+
        PhysicalKernel.integer access (buffers.input .rows) m
          (layout p h.val*p.rowStride+(r.val*16+i.val)/16)+v.val : Int).toNat*16+n=
    PhysicalKernel.integer access (buffers.input .cols) m
      ((layout p h.val*p.colStride : Nat)+
        PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val)+v.val : Int).toNat*16+n
  rw [row_div]

variable (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
  (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store)
  (q k v out : Host.Tensor) (rowCount colCount : Nat)
  (abi : PhysicalKernel.ABI p batches access buffers m q k v out rowCount colCount)
include abi

theorem query_backing (r : Row p) (h : Head p batches) (visit : Fin visits) (block : Fin p.blocks)
    (i d : Fin 16)
    (active : inputActive p batches visits (2 : Fin 5)
      (snapshot access zero decode p batches visits buffers m) (r,h,visit,block,i,d,PUnit.unit)) :
    inputOffset p batches visits (2 : Fin 5)
        (snapshot access zero decode p batches visits buffers m) (r,h,visit,block,i,d,PUnit.unit)<
      capacity (buffers.input .query).storage := by
  let id : PhysicalKernel.Program := ⟨r.val,h.val,block.val⟩
  have valid : id.Valid p batches := ⟨r.isLt,h.isLt,block.isLt⟩
  have allocated := Host.query_access (PhysicalKernel.arguments p batches access buffers m q k v)
    (PhysicalKernel.ABI.contract access p batches buffers m q k v out rowCount colCount abi)
    (id.lane (i,d,PUnit.unit)) (PhysicalKernel.lane_valid p batches _ _ id valid (i,d,PUnit.unit))
    block.val d.val block.isLt d.isLt active.2
  change BlockSparseAttn.Safety.Allocated q.storage _ at allocated
  rw [← abi.inputs.1] at allocated
  have bound := allocated_bound _ _ allocated
  simp only [Int.toNat_natCast] at bound
  simpa only [PhysicalKernel.arguments,layoutOf,PhysicalKernel.Program.lane,id,laneWork,address,
    inputOffset,batch,head,Nat.mul_add,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,
    Nat.add_comm,Nat.add_left_comm] using bound

theorem rows_backing (r : Row p) (h : Head p batches) (e : Fin 2)
    (blocks : 0<p.blocks) :
    layout p h.val*p.rowStride+r.val+e.val<capacity (buffers.input .rows).storage := by
  let id : PhysicalKernel.Program := ⟨r.val,h.val,0⟩
  have valid : id.Valid p batches := ⟨r.isLt,h.isLt,blocks⟩
  have endpoints := (PhysicalKernel.ABI.contract access p batches buffers m q k v out rowCount colCount abi).data.rowEndpoints
    (id.lane (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit))
    (PhysicalKernel.lane_valid p batches _ _ id valid _)
  have firstBound := allocated_bound _ _ endpoints.2.2.1
  have lastBound := allocated_bound _ _ endpoints.2.2.2
  norm_cast at lastBound
  rcases e with ⟨e,he⟩
  interval_cases e
  · simpa only [PhysicalKernel.arguments,layoutOf,PhysicalKernel.Program.lane,id,laneWork,
      layout,head,Int.toNat_natCast,Nat.add_zero] using firstBound
  · simpa only [PhysicalKernel.arguments,layoutOf,PhysicalKernel.Program.lane,id,laneWork,
      layout,head,Int.toNat_natCast] using lastBound

theorem columns_backing (r : Row p) (h : Head p batches) (visit : Fin visits)
    (active : inputActive p batches visits (1 : Fin 5)
      (snapshot access zero decode p batches visits buffers m) (r,h,visit,PUnit.unit)) :
    inputOffset p batches visits (1 : Fin 5)
        (snapshot access zero decode p batches visits buffers m) (r,h,visit,PUnit.unit)<
      capacity (buffers.input .cols).storage := by
  let id : PhysicalKernel.Program := ⟨r.val,h.val,0⟩
  have valid : id.Valid p batches := ⟨r.isLt,h.isLt,active.1⟩
  have countEq := snapshot_count access zero decode p batches visits buffers m r h (⟨0,by decide⟩ : Fin 16)
  have visitLive : visit.val<(physicalLayout access p batches buffers m).csr.count
      (physicalLayout access p batches buffers m).shape (head p h.val) (r.val*16+0) :=
    countEq.symm ▸ active.2
  have allocated := (PhysicalKernel.ABI.contract access p batches buffers m q k v out rowCount colCount abi).data.columns
    (id.lane (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit))
    (PhysicalKernel.lane_valid p batches _ _ id valid _) visit.val visitLive
  have bound := allocated_bound _ _ allocated
  dsimp only [PhysicalKernel.arguments,layoutOf,PhysicalKernel.Program.lane,id,laneWork,
    CSR.slot,CSR.first,PhysicalKernel.metadata] at bound
  change ((layout p h.val*p.colStride : Nat)+
    PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val)+visit.val : Int).toNat<_
  have rowZero : r.val*16/16=r.val := by omega
  simpa only [layout,head,Int.toNat_natCast,Nat.add_zero,rowZero,
    Int.natCast_mul,Int.add_assoc] using bound

theorem column_nonnegative (r : Row p) (h : Head p batches) (visit : Fin visits)
    (blocks : 0<p.blocks)
    (active : visit.val<count (snapshot access zero decode p batches visits buffers m) r h) :
    0≤column (snapshot access zero decode p batches visits buffers m) r h visit := by
  let id : PhysicalKernel.Program := ⟨r.val,h.val,0⟩
  have valid : id.Valid p batches := ⟨r.isLt,h.isLt,blocks⟩
  have live : visit.val<visitCount p (PhysicalKernel.state access zero buffers m id) := active
  have nonnegative := PhysicalKernel.column_nonnegative access zero p batches buffers m q k v out rowCount colCount abi id valid visit.val live
  change 0≤PhysicalKernel.integer access (buffers.input .cols) m
    ((layout p h.val*p.colStride : Nat)+
    PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val)+visit.val : Int).toNat
  simpa only [csrColumn,KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
    PhysicalKernel.state,id,PhysicalKernel.Program.pids,layout,head,Cell.read_same,
    ite_false,ite_true,Int.natCast_mul,Int.add_assoc] using nonnegative

theorem key_value_backing (r : Row p) (h : Head p batches) (visit : Fin visits)
    (block : Fin p.blocks) (d n : Fin 16)
    (active : visit.val<count (snapshot access zero decode p batches visits buffers m) r h)
    (enabled : p.evenN=Bool.true ∨ (n.val : Int)+column (snapshot access zero decode p batches visits buffers m) r h visit*16<p.length) :
    inputOffset p batches visits (3 : Fin 5)
        (snapshot access zero decode p batches visits buffers m) (r,h,visit,block,d,n,PUnit.unit)<
      capacity (buffers.input .key).storage ∧
    inputOffset p batches visits (4 : Fin 5)
        (snapshot access zero decode p batches visits buffers m) (r,h,block,visit,n,d,PUnit.unit)<
      capacity (buffers.input .value).storage := by
  let id : PhysicalKernel.Program := ⟨r.val,h.val,block.val⟩
  have valid : id.Valid p batches := ⟨r.isLt,h.isLt,block.isLt⟩
  have countEq := snapshot_count access zero decode p batches visits buffers m r h (⟨0,by decide⟩ : Fin 16)
  have visitLive : visit.val<(physicalLayout access p batches buffers m).csr.count
      (physicalLayout access p batches buffers m).shape (head p h.val) (r.val*16+0) := countEq.symm ▸ active
  have keyEq := snapshot_key access zero decode p batches visits buffers m r h (⟨0,by decide⟩ : Fin 16) visit n.val
  have keyEnabled : (physicalLayout access p batches buffers m).evenN=Bool.true ∨
      (physicalLayout access p batches buffers m).csr.key (physicalLayout access p batches buffers m).shape
      (head p h.val) (r.val*16+0) visit.val n.val<(physicalLayout access p batches buffers m).shape.length := by
    rw [keyEq]
    simpa only [physicalLayout,layoutOf,Int.add_comm] using enabled
  obtain ⟨ka,va⟩ := (PhysicalKernel.ABI.contract access p batches buffers m q k v out rowCount colCount abi).data.keys
    (id.lane (⟨0,by decide⟩,d,PUnit.unit))
    (PhysicalKernel.lane_valid p batches _ _ id valid _) visit.val visitLive
    block.val block.isLt d.val d.isLt n.val n.isLt keyEnabled
  change BlockSparseAttn.Safety.Allocated k.storage _ at ka
  change BlockSparseAttn.Safety.Allocated v.storage _ at va
  rw [← abi.inputs.2.1] at ka
  rw [← abi.inputs.2.2] at va
  have kb := allocated_bound _ _ ka
  have vb := allocated_bound _ _ va
  have nonnegative := column_nonnegative access zero decode p batches visits buffers m q k v out rowCount colCount abi
    r h visit (Nat.zero_lt_of_lt block.isLt) active
  have colCast := Int.toNat_of_nonneg nonnegative
  change ((batch p h.val*p.kb+kv p h.val*p.kh : Nat)+
    (physicalLayout access p batches buffers m).csr.key
      (physicalLayout access p batches buffers m).shape (head p h.val) (r.val*16+0) visit.val n.val*(p.kn:Int)+
    ((block.val*16+d.val)*p.kd : Nat)).toNat<_ at kb
  change ((batch p h.val*p.vb+kv p h.val*p.vh : Nat)+
    (physicalLayout access p batches buffers m).csr.key
      (physicalLayout access p batches buffers m).shape (head p h.val) (r.val*16+0) visit.val n.val*(p.vn:Int)+
    ((block.val*16+d.val)*p.vd : Nat)).toNat<_ at vb
  rw [keyEq] at kb vb
  have col : column (snapshot access zero decode p batches visits buffers m) r h visit=
      (column (snapshot access zero decode p batches visits buffers m) r h visit).toNat := colCast.symm
  rw [col] at kb vb
  norm_cast at kb vb
  constructor
  · change batch p h.val*p.kb+kv p h.val*p.kh+n.val*p.kn+d.val*p.kd+
        (column (snapshot access zero decode p batches visits buffers m) r h visit*16*p.kn).toNat+
        block.val*16*p.kd<_
    rw [col]
    norm_cast
    simpa only [Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using kb
  · change batch p h.val*p.vb+kv p h.val*p.vh+n.val*p.vn+(block.val*16+d.val)*p.vd+
        (column (snapshot access zero decode p batches visits buffers m) r h visit*16*p.vn).toNat<_
    rw [col]
    norm_cast
    simpa only [Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using vb

theorem physical_geometry : Geometry p :=
  ⟨abi.header.shape.2.1,PhysicalKernel.abi_stride p batches access buffers m q k v out rowCount colCount abi,
    abi.parameters⟩

theorem output_backing (j : OutIndex p batches) (active : live j) :
    outputOffset p batches j<capacity buffers.output.storage := by
  have geometry := physical_geometry access p batches buffers m q k v out rowCount colCount abi
  have bound : outputOffset p batches j<batches*p.heads*p.length*(16*p.blocks) := by
    rw [output_target p batches geometry]
    exact target_bound (PhysicalKernel.shape p batches) _
      (PhysicalKernel.coordinate_active p batches (programOf j) (program_valid j) geometry.1 (laneOf j) active)
  have extent := abi.outputExtent
  rw [← abi.outputStorage] at extent
  have allocated : BlockSparseAttn.Safety.Allocated buffers.output.storage (outputOffset p batches j) := by
    refine ⟨Int.natCast_nonneg _,?_⟩
    simp only [Int.toNat_natCast]
    omega
  simpa only [Int.toNat_natCast] using allocated_bound _ _ allocated

/-- The stronger pre-existing Tensor ABI supplies backing for all declared
finite reads and writes. RawABI remains unchanged and is not silently
strengthened by this bridge. -/
theorem physical_in_bounds (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α) :
    (io a native encode observe p batches visits).InBounds (bufferBounds buffers)
      (snapshot access zero decode p batches visits buffers m) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro region other
    simp only [io,List.mem_cons,List.not_mem_nil,or_false,not_or] at other
    simp only [bufferBounds,other.1,other.2.1,other.2.2.1,other.2.2.2.1,
      other.2.2.2.2.1,other.2.2.2.2.2,ite_false]
  · intro port j active
    rcases port with ⟨port,limit⟩
    change port<5 at limit
    interval_cases port
    · rcases j with ⟨r,h,e,⟨⟩⟩
      exact rows_backing access p batches buffers m q k v out rowCount colCount abi r h e active
    · rcases j with ⟨r,h,visit,⟨⟩⟩
      exact columns_backing access zero decode p batches visits buffers m q k v out rowCount colCount abi r h visit active
    · rcases j with ⟨r,h,visit,block,i,d,⟨⟩⟩
      exact query_backing access zero decode p batches visits buffers m q k v out rowCount colCount abi r h visit block i d active
    · rcases j with ⟨r,h,visit,block,d,n,⟨⟩⟩
      exact (key_value_backing access zero decode p batches visits buffers m q k v out rowCount colCount abi r h visit block d n active.1 active.2).1
    · rcases j with ⟨r,h,block,visit,n,d,⟨⟩⟩
      exact (key_value_backing access zero decode p batches visits buffers m q k v out rowCount colCount abi r h visit block d n active.1 active.2).2
  · intro port j active
    have portZero : port=(0 : Fin 1) := Fin.eq_zero port
    subst port
    exact output_backing access p batches buffers m q k v out rowCount colCount abi j active
  · intro port
    exact port.elim0

#axiomsClean physical_in_bounds
#axiomsClean physical_geometry
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
