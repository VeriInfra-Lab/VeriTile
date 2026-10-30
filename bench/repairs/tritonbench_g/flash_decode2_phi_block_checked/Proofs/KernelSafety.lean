import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelSafety
open VeriTile.Triton FP.Structural FlashDecode2 KernelModel KernelTensor
noncomputable section
set_option maxHeartbeats 8000000

theorem facts (c : Call) (t : Arguments c) (h : Checked.metadataChecks c.val) :
    ∀ o ∈ Checked.tensors c.val,NativeSafety.PointerFacts o := by
  intro o member
  have contiguous := (h.1 o member).2.2.2.1
  have zero := (h.1 o member).2.2.2.2.1
  simp only [Checked.tensors,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl
  · rw [←t.midObserved] at contiguous zero ⊢; exact NativeSafety.fp_facts t.mid contiguous zero
  · rw [←t.logObserved] at contiguous zero ⊢; exact NativeSafety.fp_facts t.log contiguous zero
  · rw [←t.lengthsObserved] at contiguous zero ⊢; exact NativeSafety.length_facts t.lengths contiguous zero
  · rw [←t.outputObserved] at contiguous zero ⊢; exact NativeSafety.fp_facts t.output contiguous zero

theorem pointers (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) : NativeSafety.AllAccess c.val := by
  intro active o member
  exact NativeSafety.access c.val o (facts c t h.metadata o member) ((h.launch active).1 o member)

/-- All active masked loads/stores are within the supplied element domains; promoted
INT64 byte offsets are exact for every admitted dimension, length and PID. -/
theorem offsets {α : Type} [Inhabited α] (observe : RegionName → Nat → Native.Word → α)
    (c : Call) (heap : Heap) (other : State α) (h : InputContract c (lengths c heap))
    (row j : Nat) (hr : row < rows c)
    (hj : j < iterations c (lengths c heap) row) (k : TileIndex [Checked.block c.val])
    (hk : lane k  <  Checked.width c.val) :
    let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (lift observe c heap other)
    s.pids 0 < Checked.batches c.val ∧ s.pids 1 < Checked.heads c.val ∧
    j < Checked.capacity c.val ∧
    eoff s (Checked.heads c.val) (Checked.capacity c.val) j < (Checked.geometry c.val).logSize ∧
    voff s (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k < (Checked.geometry c.val).midSize ∧
    ooff s (Checked.heads c.val) (Checked.width c.val) k < size c ∧
    4*eoff s (Checked.heads c.val) (Checked.capacity c.val) j < 2^63 ∧
    4*voff s (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k < 2^63 ∧
    4*ooff s (Checked.heads c.val) (Checked.width c.val) k < 2^63 := by
  let g := Checked.geometry c.val
  have valid : g.Valid := h.metadata.2.2.2.2.2.2.2.1
  have counted := count_bound c _ h (lift observe c heap other) (lift_length observe c heap other) row hr
  have js : j < g.capacity := Nat.lt_of_lt_of_le hj counted.2.2
  have kd : lane k < g.dimension := hk
  have rh : row < g.rows := hr
  have hh : 0 < g.heads := by dsimp [Geometry.rows] at rh; nlinarith
  have hb := (Nat.div_lt_iff_lt_mul hh).mpr rh
  have hd := Nat.mod_lt row hh
  have logBound := log_offset_bound g row j rh js
  have outBound := output_offset_bound g row (lane k) rh kd
  have midBound : row*g.capacity*g.dimension+j*g.dimension+lane k < g.midSize := by
    have subrow : row*g.capacity+j < g.rows*g.capacity := by nlinarith
    dsimp [Geometry.midSize]
    nlinarith
  have sizes := valid.2.2.2.2.2.2.2.2
  dsimp only
  simp only [row_log_offset,row_mid_offset,Grid.output_offset]
  refine ⟨hb,hd,js,logBound,midBound,outBound,?_,?_,?_⟩
  · exact byte_address_fits g.logSize g.laneBlock _ (sizes _ (by simp))
      (Nat.lt_of_lt_of_le logBound (Nat.le_add_right _ _))
  · exact byte_address_fits g.midSize g.laneBlock _ (sizes _ (by simp))
      (Nat.lt_of_lt_of_le midBound (Nat.le_add_right _ _))
  · exact byte_address_fits g.size g.laneBlock _ (sizes _ (by simp))
      (Nat.lt_of_lt_of_le outBound (Nat.le_add_right _ _))

theorem length_add (c : Call) (heap : Heap) (h : InputContract c (lengths c heap))
    (row : Nat) (hr : row < rows c) :
    (BitVec.ofNat 64 ((lengths c heap (row/Checked.heads c.val)).toNat+c.val.blockSeq-1)).toInt =
      Int.ofNat ((lengths c heap (row/Checked.heads c.val)).toNat+c.val.blockSeq-1) := by
  have hh : 0 < Checked.heads c.val := by unfold rows at hr; nlinarith
  have hb := (Nat.div_lt_iff_lt_mul hh).mpr hr
  have active : Checked.active c.val := ⟨Nat.lt_of_le_of_lt (Nat.zero_le _) hb,hh⟩
  have length := (h.contents active _ (List.mem_range.mpr hb)).2.2
  have bound : (lengths c heap (row/Checked.heads c.val)).toNat ≤ 2^31-1 := by omega
  exact Bounds.signed64 _ (length_add_fits _ _ bound h.metadata.2.2.2.2.2.2.2.1.2.2.2.2.1)

/-- Padded lanes form exact INT64 offsets even when their memory operations
are masked. The separate pointer theorem includes their uint64 base addition. -/
theorem formed {α : Type} [Inhabited α] (observe : RegionName → Nat → Native.Word → α)
    (c : Call) (heap : Heap) (other : State α) (h : InputContract c (lengths c heap))
    (row j : Nat) (hr : row < rows c) (hj : j < iterations c (lengths c heap) row)
    (k : TileIndex [Checked.block c.val]) :
    let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (lift observe c heap other)
    4*voff s (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k < 2^63 ∧
    4*ooff s (Checked.heads c.val) (Checked.width c.val) k < 2^63 := by
  let g := Checked.geometry c.val
  have valid : g.Valid := h.metadata.2.2.2.2.2.2.2.1
  have counted := count_bound c _ h (lift observe c heap other) (lift_length observe c heap other) row hr
  have js : j < g.capacity := Nat.lt_of_lt_of_le hj counted.2.2
  have mid := formed_mid_bound g row j (lane k) hr js k.1.isLt
  have out := formed_output_bound g row (lane k) hr k.1.isLt
  have sizes := valid.2.2.2.2.2.2.2.2
  dsimp only
  simp only [row_mid_offset,Grid.output_offset]
  constructor
  · apply byte_address_fits g.midSize g.laneBlock _ (sizes _ (by simp))
    simpa only [Nat.mul_assoc] using mid
  · exact byte_address_fits g.size g.laneBlock _ (sizes _ (by simp)) out

#axiomsClean formed
#axiomsClean pointers
#axiomsClean offsets
#axiomsClean length_add
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelSafety
