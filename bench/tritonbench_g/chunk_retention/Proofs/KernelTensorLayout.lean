import bench.tritonbench_g.chunk_retention.Proofs.KernelNativePacket
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelTensor
open VeriTile Triton FP.Structural KernelPacket
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 5000000

def regionShape (c : Shape) (r : RegionName) : List Nat := match r.name with
  | "q" | "k" | "dq" | "dk"=>[c.B,c.H,c.T,c.K]
  | "v" | "incoming" | "o"=>[c.B,c.H,c.T,c.V]
  | "h" | "dh"=>[c.B,c.H,c.chunks*c.K,c.V]
  | "dv"=>[c.keyBlocks,c.B,c.H,c.T,c.V]
  | _=>[c.B,c.H,c.K,c.V]

/-- Observed Tensor objects, standard TensorLaws, and resolved contiguous
metadata. Storage offsets are arbitrary; no result or allocation is assumed. -/
structure View (c : Shape) (r : RegionName) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape=regionShape c r
  contiguous : tensor.metadata.isContiguous=Bool.true
  resolved : tensor.metadata.isNeg=Bool.false
abbrev Binding (c : Shape) := (r : RegionName)→View c r
def address (v : View c r) (i : Nat) := v.tensor.metadata.dataPtr.toNat+4*i
def logical (read : Nat→β) (v : View c r) (i : Nat) : β :=
  read (v.tensor.storage.address (v.tensor.physicalIndex i))

theorem coordinates (v : View c r) (i : Nat) (hi : i<v.tensor.metadata.numel) :
    v.tensor.storage.address (v.tensor.physicalIndex i)=address v i := by
  rw [v.laws.contiguous v.contiguous i hi]
  have hp:=v.laws.pointer (by omega)
  have hc:=VeriTile.Torch.FP32.contiguous_capacity v.tensor v.laws v.contiguous (by omega)
  have hs:=v.tensor.storage.spanFits
  have hb : v.tensor.storage.address v.tensor.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have he:=congrArg BitVec.toNat hp
  simp only [VeriTile.Torch.FP32.Tensor.dataPtr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hb] at he
  unfold address VeriTile.Torch.FP32.Storage.address
  rw [he]
  simp only [VeriTile.Torch.FP32.Storage.address]
  ring

theorem address_inside (v : View c r) (i : Nat) (hi : i<v.tensor.metadata.numel) :
    v.tensor.storage.base≤address v i ∧ address v i<v.tensor.storage.base+4*v.tensor.storage.elements := by
  rw [←coordinates v i hi]
  have hb:=v.laws.inBounds i hi
  unfold VeriTile.Torch.FP32.Storage.address
  omega

theorem digit_bound (rows width row col : Nat) (hr : row<rows) (hc : col<width) :
    row*width+col<rows*width := by
  calc
    row*width+col<row*width+width:=Nat.add_lt_add_left hc _
    _=(row+1)*width:=by ring
    _≤rows*width:=Nat.mul_le_mul_right width (Nat.succ_le_of_lt hr)

/-- Bounds follow from the quoted masks and the legal PID, including the
partial-DV leading key-block dimension and its actual n_bh grid metadata. -/
theorem observation_bound (a : Call) (s : State α) (legal : Legal a s)
    (o : Observation a.shape) (ho : Active a s o) (v : View a.shape o.region) :
    o.offset s<v.tensor.metadata.numel := by
  have hh:=legal.1
  cases o with
  | history n i=>
    have hn:=ho.2.1
    have hi:=ho.2.2
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hn) hi.1) hi.2
    simpa [Observation.offset,ForwardHSource.historyAddress,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
  | finalState i=>
    have hi:=ho.2.2
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hi.1) hi.2
    simpa [Observation.offset,ForwardHSource.stateAddress,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
  | output i=>
    have hi:=ho.2
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hi.1) hi.2
    simpa [Observation.offset,ForwardOSource.outputAddress,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
  | dhistory n i=>
    have hn : BackwardHSource.chunk a.shape n<a.shape.chunks := by
      have h:=ho.2.1
      dsimp [BackwardHSource.chunk]; omega
    have hi:=ho.2.2
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hn) hi.1) hi.2
    simpa [Observation.offset,BackwardHSource.historyAddress,ForwardHSource.historyAddress,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
  | query i=>
    have hi:=ho.2
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hi.1) hi.2
    simpa [Observation.offset,BackwardQKVSource.keyAddress,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
  | key i=>
    have hi:=ho.2
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hi.1) hi.2
    simpa [Observation.offset,BackwardQKVSource.keyAddress,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
  | valuePart f i=>
    have hi:=ho.2.2
    have hl:=legal.2.2
    rw [ho.1] at hl
    have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ (digit_bound _ _ _ _ hl.1 hh) hi.1) hi.2
    simpa [Observation.offset,BackwardQKVSource.partialAddress,legal.2.1,VeriTile.Torch.TensorMetadata.numel,
      v.shape,Observation.region,regionShape,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb

theorem footprint_bound (a : Call) (s : State α) (legal : Legal a s) (views : Binding a.shape)
    (r : RegionName) (off : Nat) (h : Footprint a s r off) : off<(views r).tensor.metadata.numel := by
  obtain ⟨o,ho,rfl,rfl⟩:=h
  exact observation_bound a s legal o ho (views o.region)

/-- Every ordinary row-major input coordinate binds to its actual Tensor data
pointer. These bounds cover all active q/k/v/incoming/history/state loads. -/
theorem input_matrix_bound (c : Shape) (v : View c r) (rows cols : Nat)
    (hs : regionShape c r=[c.B,c.H,rows,cols])
    (head row col : Nat) (hh : head<c.B*c.H) (hr : row<rows) (hc : col<cols) :
    head*(rows*cols)+row*cols+col<v.tensor.metadata.numel := by
  have hb:=digit_bound _ _ _ _ (digit_bound _ _ _ _ hh hr) hc
  simpa [VeriTile.Torch.TensorMetadata.numel,v.shape,hs,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb

theorem native_logical (read : Nat→BitVec 32) (v : View c r) (i : Nat) :
    (VeriTile.Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) v.tensor v.laws i).bits=logical read v i := by
  simp [VeriTile.Torch.FP32.logicalValue,v.resolved,logical]

#axiomsClean coordinates
#axiomsClean observation_bound
#axiomsClean footprint_bound
#axiomsClean input_matrix_bound
#specNonCircular View avoiding [run,FP.Structural.exec,KernelPacket.realTarget,KernelPacket.nativeTarget]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelTensor
