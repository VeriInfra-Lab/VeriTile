import bench.repairs.tritonbench_g.matmul_dequant_int4.Kernels
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
set_option maxHeartbeats 4000000

theorem ceil_cover (a b : Nat) (hb : 0<b) : a≤ceil a b*b := by
  have hd := Nat.div_add_mod (a+b-1) b
  have hr := Nat.mod_lt (a+b-1) hb
  unfold ceil
  rw [Nat.mul_comm b] at hd
  omega
theorem ceil_positive (a b : Nat) (ha : 0<a) (hb : 0<b) : 0<ceil a b := by
  have h := ceil_cover a b hb
  by_contra hz
  have he : ceil a b=0 := by omega
  rw [he,Nat.zero_mul] at h
  omega
theorem div_coordinate (a b c : Nat) (hb : 0<b) (hc : c<b) : (a*b+c)/b=a := by
  apply Nat.div_eq_of_lt_le <;> nlinarith
theorem ceil_of_dvd (a b : Nat) (hb : 0<b) (h : b∣a) : ceil a b=a/b := by
  obtain ⟨q,rfl⟩ := h
  unfold ceil
  have hh : b*q+b-1=q*b+(b-1) := by rw [Nat.mul_comm b q];omega
  rw [hh,div_coordinate q b (b-1) hb (by omega),Nat.mul_div_cancel_left q hb]
theorem rowMajor_bound (rows columns r c : Nat) (hr : r<rows) (hc : c<columns) :
    r*columns+c<rows*columns := by nlinarith
theorem address_injective (columns : Nat) (a b : Work)
    (ha : a.col<columns) (hb : b.col<columns)
    (h : a.row*columns+a.col=b.row*columns+b.col) : a=b := by
  have hn : 0<columns := by omega
  have hd := congrArg (fun k => k/columns) h
  have hm := congrArg (fun k => k%columns) h
  simp [div_coordinate _ _ _ hn ha,div_coordinate _ _ _ hn hb] at hd
  simp [Nat.add_mod,Nat.mod_eq_of_lt ha,Nat.mod_eq_of_lt hb] at hm
  cases a;cases b;simp_all

theorem active_addresses (s : Shape) (hs : s.Valid) (r c k : Nat)
    (hr : r<s.m) (hc : c<s.n) (hk : k<s.k) :
    aAddress s r k<s.m*s.k ∧ qAddress s k c<(s.k/8)*s.n ∧
    sAddress s k c<(s.k/s.group)*s.n ∧ zAddress s k c<(s.k/s.group)*(s.n/8) ∧
    cAddress s r c<s.m*s.n := by
  have hk8 : k/8<s.k/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr (by simpa [Nat.div_mul_cancel hs.k8] using hk)
  have hkg : k/s.group<s.k/s.group := (Nat.div_lt_iff_lt_mul hs.gpos).mpr (by simpa [Nat.div_mul_cancel hs.groups] using hk)
  have hn8 : c/8<s.n/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr (by simpa [Nat.div_mul_cancel hs.n8] using hc)
  exact ⟨rowMajor_bound _ _ _ _ hr hk,rowMajor_bound _ _ _ _ hk8 hc,
    rowMajor_bound _ _ _ _ hkg hc,rowMajor_bound _ _ _ _ hkg hn8,rowMajor_bound _ _ _ _ hr hc⟩

/-- These are the original unmasked fused kernel's stated K/N/group conditions.
They apply to its raw ABI; the s1 host never calls this kernel. -/
structure FusedDomain (s : Shape) (t : Tile) (noGroups : Bool) : Prop where
  shape : s.Valid
  tile : t.Valid
  fullK : t.bk∣s.k
  fullN : t.bn∣s.n
  fullGroup : t.bk∣s.group
  noGroupsEq : noGroups=true ↔ s.group=s.k

theorem fused_k_bound (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (block lane : Nat) (hb : block<blocksK s t) (hl : lane<t.bk) :
    block*t.bk+lane<s.k := by
  have hK : blocksK s t=s.k/t.bk := ceil_of_dvd _ _ h.tile.bkpos h.fullK
  have he : s.k/t.bk*t.bk=s.k := Nat.div_mul_cancel h.fullK
  have bound := rowMajor_bound (s.k/t.bk) t.bk block lane (hK ▸ hb) hl
  simpa [he] using bound

theorem grouped_k_identity (group bk block lane : Nat) (hg : 0<group) (hb : 0<bk)
    (hd : bk∣group) (hl : lane<bk) :
    (block*bk+lane)/group=block/(group/bk) := by
  have he : group/bk*bk=group := Nat.div_mul_cancel hd
  have hq : 0<group/bk := by nlinarith
  have hlo := Nat.div_mul_le_self block (group/bk)
  have hhi := Nat.lt_mul_div_succ block hq
  have lo := Nat.mul_le_mul_right bk hlo
  have hi := Nat.mul_lt_mul_of_pos_right hhi hb
  apply Nat.div_eq_of_lt_le <;> nlinarith

theorem fused_group_correct (s : Shape) (t : Tile) (ng : Bool) (h : FusedDomain s t ng)
    (block lane : Nat) (hb : block<blocksK s t) (hl : lane<t.bk) :
    fusedGroup s t ng block=(block*t.bk+lane)/s.group := by
  unfold fusedGroup
  cases ng with
  | true =>
    change 0=(block*t.bk+lane)/s.group
    have hg := h.noGroupsEq.mp rfl
    rw [hg,Nat.div_eq_of_lt (fused_k_bound s t true h block lane hb hl)]
  | false =>
    change block/(s.group/t.bk)=(block*t.bk+lane)/s.group
    exact (grouped_k_identity s.group t.bk block lane h.shape.gpos h.tile.bkpos h.fullGroup hl).symm

theorem bk8 (t : Tile) (h : t.Valid) : 8∣t.bk := dvd_trans (by decide : 8∣16) h.bk16
theorem packed_advance (t : Tile) (ht : t.Valid) (block lane n : Nat) :
    (lane/8)*n+block*((t.bk/8)*n)=((block*t.bk+lane)/8)*n := by
  obtain ⟨q,hq⟩ := bk8 t ht
  rw [hq]
  have he : block*(8*q)+lane=lane+(block*q)*8 := by ring
  rw [he]
  simp only [Nat.add_mul_div_right _ _ (by decide : 0<8),Nat.mul_div_cancel_left _ (by decide : 0<8)]
  ring
theorem shift_period (t : Tile) (ht : t.Valid) (block lane : Nat) :
    (block*t.bk+lane)%8=lane%8 := by
  have h := Nat.mod_eq_zero_of_dvd (bk8 t ht)
  simp [Nat.add_mod,Nat.mul_mod,h]

/-- Independent padded envelopes include inactive standalone load/store pointers
and the fused pointers after their final increment. -/
structure IndexDomain (s : Shape) where
  paddedM : Nat
  paddedN : Nat
  paddedK : Nat
  rows : s.m≤paddedM
  columns : s.n≤paddedN
  inner : s.k≤paddedK
  scalar : max paddedM (max paddedN (max paddedK s.group))<2^31
  a : paddedM*s.k+paddedK<2^31
  q : (paddedK/8+1)*s.n+paddedN<2^31
  scale : (paddedK/s.group+1)*s.n+paddedN<2^31
  zero : (paddedK/s.group+1)*(s.n/8)+paddedN/8<2^31
  output : paddedM*s.n+paddedN<2^31
  dequant : paddedK*s.n+paddedN<2^31

theorem formed_addresses (s : Shape) (d : IndexDomain s) (r c k : Nat)
    (hr : r<d.paddedM) (hc : c<d.paddedN) (hk : k≤d.paddedK) :
    aAddress s r k<2^31 ∧ qAddress s k c<2^31 ∧ sAddress s k c<2^31 ∧
    zAddress s k c<2^31 ∧ cAddress s r c<2^31 ∧ cAddress s k c<2^31 := by
  have hq := Nat.div_le_div_right hk (c:=8)
  have hg := Nat.div_le_div_right hk (c:=s.group)
  have hn := Nat.div_le_div_right (show c≤d.paddedN by omega) (c:=8)
  have am := Nat.mul_le_mul_right s.k (show r≤d.paddedM by omega)
  have qm := Nat.mul_le_mul_right s.n hq
  have sm := Nat.mul_le_mul_right s.n hg
  have zm := Nat.mul_le_mul_right (s.n/8) hg
  have cm := Nat.mul_le_mul_right s.n (show r≤d.paddedM by omega)
  have dm := Nat.mul_le_mul_right s.n hk
  have ha := d.a;have hq' := d.q;have hs := d.scale
  have hz := d.zero;have ho := d.output;have hd := d.dequant
  simp only [Nat.add_mul,Nat.one_mul] at hq' hs hz
  dsimp [aAddress,qAddress,sAddress,zAddress,cAddress]
  constructor
  · omega
  constructor
  · omega
  constructor
  · omega
  constructor
  · omega
  constructor <;> omega

theorem signed32_exact (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n:Int) := by
  rw [BitVec.toInt_eq_toNat_of_lt]
  · simp;omega
  · simp [Nat.mod_eq_of_lt (show n<2^32 by omega)];omega
theorem byte_offset_bound (n width byte : Nat) (h : n<2^31) (hw : width≤4) (hb : byte<width) :
    width*n+byte<2^63 := by nlinarith

#axiomsClean active_addresses
#axiomsClean fused_group_correct
#axiomsClean packed_advance
#axiomsClean shift_period
#axiomsClean formed_addresses
#axiomsClean signed32_exact
end VeriTile.Bench.Repairs.MatmulDequantInt4
