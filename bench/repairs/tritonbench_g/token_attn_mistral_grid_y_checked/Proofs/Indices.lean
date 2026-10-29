import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.Tensor

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile.Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxHeartbeats 4000000

/-- Legal axes of the actual submitted grid. This candidate makes no new
grid-product compatibility claim and introduces no grid-product guard. -/
theorem launch_bounds (m : Inputs) (read : Reader) (c : InputContract m read)
    (live : 0 < batches m) : batches m ≤ 2^31-1 ∧ heads m ≤ 65535 := by
  have limit := c.header.extents .request
  change (m .request).numel ≤ 2^31-1 ∧ _
  exact ⟨by omega,c.gridY live⟩

/-- Checked padded arithmetic, including lanes whose loads are masked. -/
theorem formed32 (m : Inputs) (read : Reader) (c : InputContract m read)
    (b h cur lane : Nat) (hb : b < batches m) (hh : h < heads m)
    (hc : cur < index m read .length b) (hl : lane < 128) :
    cur+128 < 2^31 ∧ cur+lane < 2^31 ∧ first m read b+cur+lane < 2^31 ∧
    probabilityIndex m read b h (cur+lane) < 2^31 ∧
    index m read .request b*columns m+(first m read b+cur+lane) < 2^31 := by
  obtain ⟨hr,hL,hS,_⟩ := data_bounds m read c b hb
  obtain ⟨pe,_,te⟩ := extents m c.header
  obtain ⟨lp,lt,lc⟩ := c.header.padded
  rw [pe] at lp
  rw [te] at lt
  have hm := Nat.mul_le_mul_right (tokens m) hh
  have rm := Nat.mul_le_mul_right (columns m) hr
  simp only [Nat.succ_mul] at hm rm
  have hf : first m read b ≤ columns m := by
    unfold first
    omega
  dsimp [probabilityIndex]
  omega

theorem active32 (m : Inputs) (read : Reader) (c : InputContract m read)
    (b h n d : Nat) (hb : b < batches m) (hh : h < heads m)
    (hn : first m read b+n < index m read .sequence b) (hd : d < width m) :
    valueIndex m read b h n d < 2^31 ∧ outputIndex m b h d < 2^31 := by
  have v := (value_bounds m read c b h n d hb hh hn hd).2
  have o := output_bound m read c b h d hb hh hd
  have lv := c.header.extents .value
  have lo := c.header.extents .output
  omega

/-- Every dynamic scalar passed as a source stride/group is signed32-safe,
including calls with empty probability/cache tensors and zero row lengths. -/
theorem scalar32 (m : Inputs) (read : Reader) (c : InputContract m read)
    (live : 0 < batches m) :
    tokens m < 2^31 ∧ columns m < 2^31 ∧ width m < 2^31 ∧
    heads m*width m < 2^31 ∧ kvheads m*width m < 2^31 ∧ group m < 2^31 := by
  have cols := c.header.padded.2.2
  have head := c.gridY live
  have dim := c.header.dimension
  have gp := group_shape m c.header
  have kv : kvheads m ≤ heads m := by
    have h := Nat.le_mul_of_pos_right (kvheads m) gp.1
    rwa [gp.2] at h
  have gh : group m ≤ heads m := Nat.div_le_self _ _
  have dw : width m ≤ 128 := by rcases dim with h|h|h|h <;> omega
  have hw := Nat.mul_le_mul head dw
  have kw := Nat.mul_le_mul kv (Nat.le_refl (width m))
  omega

theorem metadata_bounds (m : Inputs) (read : Reader) (c : InputContract m read)
    (r : Role) (hr : r∈[Role.request,.unusedStart,.sequence,.start,.length])
    (b : Nat) (hb : b < batches m) : b < (m r).numel ∧ b < 2^31 := by
  have sh := c.header.vectors r hr
  have size : (m r).numel=batches m := by
    simp [VeriTile.Torch.TensorMetadata.numel,sh]
  have extent := c.header.extents r
  exact ⟨by omega,by omega⟩

/-- V's offset is formed even for masked lanes, where the table load supplied
zero. Both that zero-page expression and each active gathered page fit int32. -/
theorem value_formed32 (m : Inputs) (read : Reader) (c : InputContract m read)
    (b h n d : Nat) (hb : b < batches m) (hh : h < heads m) (hd : d < width m) :
    h/group m*width m+d+
      (if first m read b+n < index m read .sequence b then page m read b n else 0)*
        (kvheads m*width m) < 2^31 := by
  by_cases active : first m read b+n < index m read .sequence b
  · simpa only [active,ite_true] using (active32 m read c b h n d hb hh active hd).1
  · have baseBound := linear_bound _ _ _ _ (group_bound m c.header h hh) hd
    have scalar := (scalar32 m read c (by omega)).2.2.2.2.1
    simp only [active,ite_false,Nat.zero_mul,Nat.add_zero]
    omega

/-- Nonnegative int32 metadata has exactly the Nat meaning used by the quoted
program; no sign-erasing premise is assumed for arbitrary words. -/
theorem signed_index (m : Inputs) (read : Reader) (r : Role) (i : Nat)
    (nonnegative : 0 ≤ raw m read r i) :
    (index m read r i : Int)=(read (base m r+4*i)).toInt := by
  exact Int.toNat_of_nonneg nonnegative

/-- Signed subtraction is in range before maximum with zero. The Nat adapter
therefore denotes precisely the source's signed sliding-window start. -/
theorem signed_window (m : Inputs) (read : Reader) (c : InputContract m read)
    (b : Nat) (hb : b < batches m) :
    -(2^31 : Int) ≤ raw m read .sequence b-m.slidingWindow ∧
    raw m read .sequence b-m.slidingWindow < 2^31 ∧
    max (raw m read .sequence b-m.slidingWindow) 0=(first m read b : Int) := by
  have seq := (c.contents b hb).2.1
  have win := c.header.window
  have col := c.header.padded.2.2
  simp only [first,index,window]
  omega

theorem signed32 (n : Nat) (small : n < 2^31) :
    (BitVec.ofNat 32 n).toInt=(n : Int) := by
  rw [BitVec.toInt_ofNat']
  apply Int.bmod_eq_of_le <;> omega

/-- Every accessed address has four-byte alignment and lies wholly inside the
64-bit pointer space. Masked pointers need no backing or dereference. -/
theorem address64 (m : Inputs) (read : Reader) (c : InputContract m read)
    (live : 0 < batches m) (r : Role) (i : Nat) (hi : i < (m r).numel) :
    (base m r+4*i)%4=0 ∧ base m r+4*i+4 ≤ 2^64 ∧
    (BitVec.ofNat 64 (base m r+4*i)).toNat=base m r+4*i := by
  have backing := (c.physical live).backing r
  have align := (c.physical live).alignment r (by omega)
  refine ⟨by simpa [Nat.add_mod] using align,by omega,?_⟩
  exact Nat.mod_eq_of_lt (by omega)

#axiomsClean launch_bounds
#axiomsClean formed32
#axiomsClean active32
#axiomsClean scalar32
#axiomsClean metadata_bounds
#axiomsClean value_formed32
#axiomsClean signed_index
#axiomsClean signed_window
#axiomsClean address64
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
