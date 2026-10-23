import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.GridExecution

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
namespace PhysicalKernel
variable {α : Type} [Inhabited α]
variable (access : Access α Store Byte) (zero : α) (p : Parameters) (batches : Nat)
  (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
  (abi : ABI p batches access buffers initial q k v out rows cols)

include abi

theorem ABI.contract : Host.Contract (arguments p batches access buffers initial q k v) :=
  CheckedCSR.contract _ abi.header abi.csr


/-- Enabled floating reads are exactly reads of the caller's current memory;
the totalized adapter's fallback is unreachable at these addresses. -/
theorem query_load (m : Store) (l : Lane) (valid : l.Valid ((arguments p batches access buffers initial q k v).layout).shape)
    (db d : Nat) (hdb : db<((arguments p batches access buffers initial q k v).layout).shape.blocks) (hd : d<((arguments p batches access buffers initial q k v).layout).shape.blockD)
    (enabled : ((arguments p batches access buffers initial q k v).layout).evenM=Bool.true ∨ (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row<((arguments p batches access buffers initial q k v).layout).shape.length) :
    Physical.inputs access.toAPI zero buffers m .q
      (address ((arguments p batches access buffers initial q k v).layout).q (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).batch (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).head (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row (db*((arguments p batches access buffers initial q k v).layout).shape.blockD+d))=
    access.read m (Physical.inputAddress (buffers.input .query)
      (address ((arguments p batches access buffers initial q k v).layout).q (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).batch (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).head (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row (db*((arguments p batches access buffers initial q k v).layout).shape.blockD+d))) := by
  have allocated := Host.query_access _ (ABI.contract access p batches buffers initial q k v out rows cols abi)
    l valid db d hdb hd enabled
  change Safety.Allocated q.storage _ at allocated
  rw [← abi.inputs.1] at allocated
  exact Physical.read_active access.toAPI zero (buffers.input .query) m _ allocated

theorem key_value_load (m : Store) (l : Lane) (valid : l.Valid ((arguments p batches access buffers initial q k v).layout).shape)
    (visit : Nat) (hv : visit<((arguments p batches access buffers initial q k v).layout).csr.count ((arguments p batches access buffers initial q k v).layout).shape (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).head (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row)
    (db d n : Nat) (hdb : db<((arguments p batches access buffers initial q k v).layout).shape.blocks) (hd : d<((arguments p batches access buffers initial q k v).layout).shape.blockD) (hn : n<((arguments p batches access buffers initial q k v).layout).shape.blockN)
    (enabled : ((arguments p batches access buffers initial q k v).layout).evenN=Bool.true ∨ ((arguments p batches access buffers initial q k v).layout).csr.key ((arguments p batches access buffers initial q k v).layout).shape (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).head
      (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row visit n<(((arguments p batches access buffers initial q k v).layout).shape.length:Int)) :
    let w:=laneWork ((arguments p batches access buffers initial q k v).layout).shape l
    let key:=((arguments p batches access buffers initial q k v).layout).csr.key ((arguments p batches access buffers initial q k v).layout).shape w.head w.row visit n
    let ki:=(signedAddress ((arguments p batches access buffers initial q k v).layout).k w.batch (kvHead ((arguments p batches access buffers initial q k v).layout).shape w.head) key (db*((arguments p batches access buffers initial q k v).layout).shape.blockD+d)).toNat
    let vi:=(signedAddress ((arguments p batches access buffers initial q k v).layout).v w.batch (kvHead ((arguments p batches access buffers initial q k v).layout).shape w.head) key (l.channelProgram*((arguments p batches access buffers initial q k v).layout).shape.blockD+d)).toNat
    Physical.inputs access.toAPI zero buffers m .k ki=access.read m (Physical.inputAddress (buffers.input .key) ki) ∧
    Physical.inputs access.toAPI zero buffers m .v vi=access.read m (Physical.inputAddress (buffers.input .value) vi) := by
  obtain ⟨ka,va⟩ := (ABI.contract access p batches buffers initial q k v out rows cols abi).data.keys
    l valid visit hv db hdb d hd n hn enabled
  change Safety.Allocated k.storage _ at ka
  change Safety.Allocated v.storage _ at va
  rw [← abi.inputs.2.1] at ka
  rw [← abi.inputs.2.2] at va
  exact ⟨Physical.read_active access.toAPI zero (buffers.input .key) m _ ka,
    Physical.read_active access.toAPI zero (buffers.input .value) m _ va⟩

omit abi in
theorem integer_load (buffer : Physical.Buffer) (m : Store) (i : Int)
    (allocated : Safety.Allocated buffer.storage i) :
    integer access buffer m i.toNat=(access.read32 m (Physical.inputAddress buffer i.toNat)).toInt :=
  if_pos allocated.2

/-- Both CSR endpoint words and every visited column word are allocated;
signed loop and pointer nodes are exact INT32 arithmetic under inspected CSR. -/
theorem csr_access : Safety.DataContract ((arguments p batches access buffers initial q k v).layout) (buffers.input .rows).storage
    (buffers.input .cols).storage k.storage v.storage :=
  (ABI.contract access p batches buffers initial q k v out rows cols abi).data

theorem data_i32 (l : Lane) (valid : l.Valid ((arguments p batches access buffers initial q k v).layout).shape) (visit : Nat)
    (hv : visit<((arguments p batches access buffers initial q k v).layout).csr.count ((arguments p batches access buffers initial q k v).layout).shape (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).head (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row)
    (db d n : Nat) (hdb : db<((arguments p batches access buffers initial q k v).layout).shape.blocks) (hd : d<((arguments p batches access buffers initial q k v).layout).shape.blockD) (hn : n<((arguments p batches access buffers initial q k v).layout).shape.blockN)
    (z : Int) (hz : z∈Safety.dataNodes ((arguments p batches access buffers initial q k v).layout) l visit db d n) :
    (BitVec.ofInt 32 z).toInt=z :=
  Safety.all_data_nodes_exact ((arguments p batches access buffers initial q k v).layout) _ _ _ _ (csr_access access p batches buffers initial q k v out rows cols abi)
    l valid visit hv db d n hdb hd hn z hz

theorem query_i32 (l : Lane) (valid : l.Valid ((arguments p batches access buffers initial q k v).layout).shape)
    (db d : Nat) (hdb : db<((arguments p batches access buffers initial q k v).layout).shape.blocks) (hd : d<((arguments p batches access buffers initial q k v).layout).shape.blockD) :
    address ((arguments p batches access buffers initial q k v).layout).q (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).batch (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).head
      (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).row (db*((arguments p batches access buffers initial q k v).layout).shape.blockD+d)<2^31 :=
  query_grid_i32 _ (ABI.contract access p batches buffers initial q k v out rows cols abi) l valid db d hdb hd

theorem output_i32 (l : Lane) (valid : l.Valid ((arguments p batches access buffers initial q k v).layout).shape) :
    (laneWork ((arguments p batches access buffers initial q k v).layout).shape l).target ((arguments p batches access buffers initial q k v).layout).shape<2^31 :=
  output_grid_i32 _ (ABI.contract access p batches buffers initial q k v out rows cols abi) l valid

theorem whole_input_frame (a : Arithmetic α) (native : Bool) :
    ∃ final,grid access a native p buffers (programs p batches) initial=some final ∧
      Stable access a.zero buffers initial final := by
  refine ⟨_,grid_success access a native p batches buffers initial q k v out rows cols abi,?_⟩
  apply run_stable access a.zero buffers _ _ abi.separate initial initial (Stable.refl ..)
  intro e member
  obtain ⟨id,hid,he⟩ := List.mem_flatMap.mp member
  exact events_bounded access a native p batches buffers initial q k v out rows cols abi id
    ((programs_mem p batches id).mp hid) e he

end PhysicalKernel
#axiomsClean PhysicalKernel.ABI.contract
#axiomsClean PhysicalKernel.query_load
#axiomsClean PhysicalKernel.key_value_load
#axiomsClean PhysicalKernel.integer_load
#axiomsClean PhysicalKernel.csr_access
#axiomsClean PhysicalKernel.data_i32
#axiomsClean PhysicalKernel.query_i32
#axiomsClean PhysicalKernel.output_i32
#axiomsClean PhysicalKernel.whole_input_frame
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
