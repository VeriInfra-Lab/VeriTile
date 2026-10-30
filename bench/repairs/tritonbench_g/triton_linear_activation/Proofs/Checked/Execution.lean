import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Machine
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Entry

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
set_option maxHeartbeats 8000000

/-- Every live output/saved logical index is within its contiguous footprint. -/
theorem output_index (r : Request) (m n : Nat) (hm : m < r.batch.rows) (hn : n < r.n) :
    outputOffset r.shape m n < r.batch.rows*r.n := by
  have positive : 1 ≤ r.n := by omega
  simp only [outputOffset,Request.shape,max_eq_left positive]
  have := Nat.mul_le_mul_right r.n (Nat.succ_le_of_lt hm)
  nlinarith

theorem cell_capacity (mode : Mode) (index count bytes : Nat) (hi : index < count)
    (capacity : byteWidth mode*count ≤ bytes) : byteWidth mode*(index+1) ≤ bytes :=
  (Nat.mul_le_mul_left (byteWidth mode) (Nat.succ_le_of_lt hi)).trans capacity

/-- Physical pointer addition for every byte of an active cell. Allocation
extent prevents unsigned wrap. Masked formed offsets are separately covered
by the source signed32/final-increment theorems in Proofs/Machine.lean. -/
theorem native_cell (backing : Backing) (mode : Mode) (index j : Nat)
    (capacity : byteWidth mode*(index+1) ≤ backing.bytes)
    (extent : backing.baseByte+backing.bytes ≤ 2^64) (byte : j < byteWidth mode) :
    (BitVec.ofNat 64 backing.baseByte + BitVec.ofNat 64 (byteWidth mode*index+j)).toNat =
      backing.baseByte+byteWidth mode*index+j ∧
    backing.covers backing.owner (backing.baseByte+byteWidth mode*index+j) := by
  have bound : byteWidth mode*index+j < backing.bytes := by
    rw [Nat.mul_add,Nat.mul_one] at capacity
    omega
  constructor
  · rw [BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
    omega
  · exact ⟨rfl,by omega,by omega⟩

theorem caller_native_cell (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (ts : Fin 4 → Observed) (binding : PhysicalBinding h layout ts) (admit : Domain r ts flag)
    (slot : Fin 4) (used : Used r slot) (index j : Nat) (hi : index < r.count slot) (byte : j < 4) :
    (BitVec.ofNat 64 (ts slot).base + BitVec.ofNat 64 (4*index+j)).toNat = (ts slot).base+4*index+j ∧
    (layout.backing (r.storage slot)).covers (ts slot).metadata.device.index ((ts slot).base+4*index+j) := by
  have count := bound_numel r ts binding.identity slot used (admit.shape slot used)
  have capacity := binding.capacity slot used
  rw [count] at capacity
  have physical := binding.backing slot used
  have cap : 4*r.count slot ≤ (layout.backing (r.storage slot)).bytes := by simpa [physical.2.2] using capacity
  have native := native_cell (layout.backing (r.storage slot)) .fp32 index j
    (cell_capacity .fp32 _ _ _ hi cap) (binding.nativeExtent _) byte
  simpa only [byteWidth,physical.1,physical.2.1] using native

theorem output_native_cell (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (ts : Fin 4 → Observed) (binding : PhysicalBinding h layout ts)
    (m n j : Nat) (hm : m < r.batch.rows) (hn : n < r.n) (byte : j < byteWidth r.mode) :
    (BitVec.ofNat 64 (layout.backing (h.storage 0)).baseByte +
      BitVec.ofNat 64 (byteWidth r.mode*outputOffset r.shape m n+j)).toNat =
      (layout.backing (h.storage 0)).baseByte+byteWidth r.mode*outputOffset r.shape m n+j ∧
    (layout.backing (h.storage 0)).covers (layout.backing (h.storage 0)).owner
      ((layout.backing (h.storage 0)).baseByte+byteWidth r.mode*outputOffset r.shape m n+j) :=
  native_cell _ r.mode _ j (cell_capacity _ _ _ _ (output_index r m n hm hn) layout.outputBytes)
    (binding.nativeExtent _) byte

/-- Admission discharges the source's actual signed32 arithmetic side
conditions for every original launch configuration. -/
theorem checked_output_machine (admit : Domain r ts flag) (c : Fin 98)
    (pid i j : Nat) (hp : pid < r.shape.programs c) (hi : i < (configs c).bm) (hj : j < (configs c).bn) :
    (affine32 (row r.shape c pid i) (max r.n 1) (col r.shape c pid j)).toInt =
      (row r.shape c pid i*max r.n 1+col r.shape c pid j : Nat) ∧
    4*(row r.shape c pid i*max r.n 1+col r.shape c pid j) < 2^63 :=
  (output_machine_exact r.shape c admit.bounds pid i j hp hi hj).2.2

#axiomsClean caller_native_cell
#axiomsClean output_native_cell
#axiomsClean checked_output_machine
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
