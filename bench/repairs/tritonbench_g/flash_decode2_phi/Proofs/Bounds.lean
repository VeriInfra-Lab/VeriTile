import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Coordinates
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Physical

namespace VeriTile.Bench.Repairs.FlashDecode2.Bounds
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

/-- Signed interpretation of every promoted nonnegative native integer. -/
theorem signed64 (n : Nat) (bound : n<2^63) : (BitVec.ofNat 64 n).toInt=(n : Int) := by
  have hn : (BitVec.ofNat 64 n).toNat=n := by
    change n%2^64=n
    exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

theorem positive_int32 (word : BitVec 32) (bound : word.toNat≤2^31-1) :
    word.toInt=(word.toNat : Int) := by
  exact BitVec.toInt_eq_toNat_of_lt (by omega)

/-- Actual loaded length, dynamic iteration, formed padded addresses, and
active memory accesses are all tied to the checked host geometry. -/
theorem source_addresses {α : Type} [Inhabited α] (q : Host.Request) (call : Host.Preparation α q)
    (row j : Nat) (hr : row<q.g.rows)
    (hj : j<count (Grid.setup q.g.batches q.g.heads row call.before) q.g.sequenceBlock)
    (k : TileIndex [q.g.laneBlock]) :
    let s := Grid.setup q.g.batches q.g.heads row call.before
    s.pids 0<q.g.batches ∧ s.pids 1<q.g.heads ∧
    j<q.g.capacity ∧
    eoff s q.g.heads q.g.capacity j<q.g.logSize ∧
    voff s q.g.heads q.g.capacity q.g.dimension j k<q.g.midSize+q.g.laneBlock ∧
    ooff s q.g.heads q.g.dimension k<q.g.size+q.g.laneBlock ∧
    (active q.g.masked q.g.dimension k →
      voff s q.g.heads q.g.capacity q.g.dimension j k<q.g.midSize ∧
      ooff s q.g.heads q.g.dimension k<q.g.size) ∧
    4*eoff s q.g.heads q.g.capacity j<2^63 ∧
    4*voff s q.g.heads q.g.capacity q.g.dimension j k<2^63 ∧
    4*ooff s q.g.heads q.g.dimension k<2^63 := by
  obtain ⟨hH,hb,hh⟩ := Host.row_indices q row hr
  have hjS : j<q.g.capacity := lt_of_lt_of_le hj (Host.row_count q call row hr).2
  have he := log_offset_bound q.g row j hr hjS
  have hm := formed_mid_bound q.g row j (lane k) hr hjS k.1.isLt
  have ho := formed_output_bound q.g row (lane k) hr k.1.isLt
  have hlog : row*q.g.capacity+j<q.g.logSize+q.g.laneBlock := by omega
  have sizes := q.valid.2.2.2.2.2.2.2.2
  dsimp only
  simp only [row_log_offset,row_mid_offset,Grid.output_offset]
  refine ⟨hb,hh,hjS,he,?_,ho,?_,?_,?_,?_⟩
  · simpa only [Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using hm
  · intro hk
    have hd := (Grid.active_iff q.g.masked q.g.dimension q.g.laneBlock q.valid.2.2.1 k).mp hk
    constructor
    · dsimp [Geometry.midSize]
      have hrow : row*q.g.capacity+j<q.g.rows*q.g.capacity := by nlinarith
      nlinarith
    · exact output_offset_bound q.g row (lane k) hr hd
  · exact byte_address_fits q.g.logSize q.g.laneBlock _ (sizes _ (by simp)) hlog
  · apply byte_address_fits q.g.midSize q.g.laneBlock _ (sizes _ (by simp))
    simpa only [Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using hm
  · exact byte_address_fits q.g.size q.g.laneBlock _ (sizes _ (by simp)) ho

/-- Concrete native byte writes preserve every input allocation at every
bounded schedule prefix, so FP32 and INT32 loads keep reading the entry bytes. -/
theorem input_bytes (I : Native.Instructions) (S : Native.Schedule) (q : Host.Request)
    (call : Host.Preparation Native.Term q) (memory : Physical.Bytes)
    (rows : List Nat) (bounded : ∀ row∈rows,row<q.g.rows)
    (r : RegionName) (input : r=call.lengths.allocation ∨ r=call.mid.allocation ∨ r=call.log.allocation)
    (b : Nat) : Physical.executeBytes I S call.out.allocation q rows call.before memory r b=memory r b := by
  by_cases empty : q.g.rows=0
  · have hs : rows=[] := by
      by_contra h
      obtain ⟨x,hx⟩ := List.exists_mem_of_ne_nil rows h
      have := bounded x hx
      omega
    simp [hs,Physical.executeBytes]
  · apply Physical.execute_frame I S call.out.allocation q rows bounded call.before memory r b
    left
    obtain ⟨hM,hE,hL,_⟩ := call.disjoint (by omega)
    rcases input with h|h|h
    · simpa [h] using Ne.symm hL
    · simpa [h] using Ne.symm hM
    · simpa [h] using Ne.symm hE

#axiomsClean signed64
#axiomsClean positive_int32
#axiomsClean source_addresses
#axiomsClean input_bytes
end
end VeriTile.Bench.Repairs.FlashDecode2.Bounds
