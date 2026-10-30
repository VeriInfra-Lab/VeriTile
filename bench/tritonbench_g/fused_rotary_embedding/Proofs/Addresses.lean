import bench.tritonbench_g.fused_rotary_embedding.Proofs.Contract

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Checked

abbrev pad (s : Shape) := s.width+s.halfBlock+s.fullBlock

theorem data_formed (s : Shape) (hs : s.Valid) (kind : Kind) (t h d : Nat)
    (ht : t<s.tokens) (hh : h<s.heads kind) (hd : d<pad s) :
    s.data kind t h d<formed s kind.input := by
  have base := Nat.mul_lt_mul_of_pos_right (radix_bound _ _ _ _ ht hh) hs.width
  cases kind <;> simp_all [Shape.data,Shape.heads,Kind.input,formed,sourceExtent,pad] <;> omega

theorem coefficient_formed (s : Shape) (t d : Nat) (sine : Bool)
    (ht : t<s.tokens) (hd : d<s.halfBlock) :
    s.coeff sine t d<formed s (if sine then .sin else .cos) := by
  have hb := Nat.mul_le_mul_right (if sine then s.sinWidth else s.cosWidth) (Nat.le_of_lt ht)
  cases sine <;> simp_all [Shape.coeff,formed,sourceExtent] <;> omega

theorem cache_formed (s : Shape) (hs : s.Valid) (b h slot d : Nat)
    (hb : b<s.blocks) (hh : h<s.headsK) (hl : slot<s.blockSize) (hd : d<pad s)
    (ordinary : s.newLayout=false → s.split=s.width) :
    sourceCacheAddress s b h slot d<formed s .kcache := by
  dsimp only [pad] at hd
  have bh := radix_bound _ _ _ _ hb hh
  have dw : s.width/s.split*s.split=s.width := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hs.divides)
  have dx : s.split≤s.width := Nat.le_of_dvd hs.width (Nat.dvd_of_mod_eq_zero hs.divides)
  have bd := Nat.mul_le_mul_right (s.blockSize*s.width) (show b*s.headsK+h+1≤s.blocks*s.headsK by omega)
  have sl := Nat.mul_lt_mul_of_pos_right hl hs.split
  have sb := Nat.mul_le_mul_left s.blockSize dx
  have dq : d/s.split*s.split≤d := Nat.div_mul_le_self _ _
  have dm : d%s.split≤d := Nat.mod_le _ _
  have dp := Nat.mul_le_mul_right s.blockSize dq
  cases layout : s.newLayout
  · have sx := ordinary layout
    simp only [sourceCacheAddress,layout,Bool.false_eq_true,↓reduceIte,formed,sourceExtent,sx,Nat.mul_zero,Nat.add_zero]
    have slw := Nat.mul_lt_mul_of_pos_right hl hs.width
    rw [sx] at dm
    nlinarith
  · simp only [sourceCacheAddress,layout,↓reduceIte,formed,sourceExtent]
    have deq : b*(s.headsK*(s.width/s.split)*s.blockSize*s.split)+
        h*((s.width/s.split)*s.blockSize*s.split)=(b*s.headsK+h)*s.blockSize*s.width := by
      calc
        _ = (b*s.headsK+h)*s.blockSize*(s.width/s.split*s.split) := by ring
        _ = _ := by rw [dw]
    rw [deq]
    nlinarith

theorem value_cache_formed (s : Shape) (hs : s.Valid) (b h slot d : Nat)
    (hb : b<s.blocks) (hh : h<s.headsK) (hl : slot<s.blockSize) (hd : d<s.fullBlock) :
    s.cache false b h slot d<formed s .vcache := by
  have base := Nat.mul_lt_mul_of_pos_right (radix_bound _ _ _ _ (radix_bound _ _ _ _ hb hh) hl) hs.width
  simp only [Shape.cache,Bool.false_and,Bool.false_eq_true,↓reduceIte,formed,sourceExtent]
  omega

theorem paired_lane_formed (s : Shape) (j : Nat) (hj : j<s.halfBlock) (second : Bool) :
    j+(if second then s.half else 0)<pad s := by
  have hhalf : s.half≤s.width := Nat.div_le_self _ _
  cases second <;> simp_all [pad] <;> omega

theorem formed_pointer (c : Call) (m : Physical.Bytes) (checked : check c m=true)
    (live : 0<c.shape.tokens) (r : Region) (hr : r∈regions) (a : Nat) (ha : a<formed c.shape r)
    (i : Fin 4) : 4*a+i.val<2^63 ∧ (c.tensor r).base+4*a+i.val<2^64 ∧ (c.tensor r).base%4=0 := by
  have hp := (live_checked c m checked live).2.1 r hr
  refine ⟨active_byte_int64 _ _ ha hp.2.2.1 i,?_,hp.1⟩
  have room := hp.2.2.2.1
  omega

#axiomsClean data_formed
#axiomsClean coefficient_formed
#axiomsClean cache_formed
#axiomsClean value_cache_formed
#axiomsClean paired_lane_formed
#axiomsClean formed_pointer
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Checked
