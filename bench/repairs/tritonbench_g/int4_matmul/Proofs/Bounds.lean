import bench.repairs.tritonbench_g.int4_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.Int4Matmul
set_option maxHeartbeats 4000000

theorem rowMajor_bound (rows columns r c : Nat) (hr : r<rows) (hc : c<columns) :
    r*columns+c<rows*columns := by nlinarith

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

/-- Includes the final pointer increment even after the last masked load. -/
def kEnvelope (s : Shape) (t : Tile) := (rounds s t+1)*t.bk*t.splits

theorem formed_k (s : Shape) (t : Tile) (ht : t.Valid) (sp block lane : Nat)
    (hsp : sp<t.splits) (hb : block≤rounds s t) (hl : lane<t.bk) :
    kIndex t sp block lane<kEnvelope s t := by
  unfold kIndex kEnvelope
  have h1 : sp+1≤t.splits := by omega
  have h2 := Nat.mul_le_mul_right t.bk h1
  have h3 := Nat.mul_le_mul_right (t.bk*t.splits) hb
  nlinarith

/-- The source pointer recurrence agrees with packed quotient addressing because
all admitted source BK values are divisible by eight. -/
theorem packed_advance (t : Tile) (ht : t.Valid) (sp block lane n : Nat) :
    ((sp*t.bk+lane)/8)*n+block*(t.bk*t.splits*n/8)=
      (kIndex t sp block lane/8)*n := by
  obtain ⟨q,hq⟩ := ht.bk8
  unfold kIndex
  rw [hq]
  have e1 : sp*(8*q)+lane=lane+(sp*q)*8 := by ring
  have e2 : 8*q*t.splits*n=(q*t.splits*n)*8 := by ring
  have e3 : sp*(8*q)+block*(8*q)*t.splits+lane=lane+(sp*q+block*q*t.splits)*8 := by ring
  rw [e1,e2,e3]
  simp only [Nat.add_mul_div_right _ _ (by decide : 0<8),Nat.mul_div_cancel _ (by decide : 0<8)]
  ring

structure IndexDomain (s : Shape) (t : Tile) : Prop where
  a : s.m*s.k+kEnvelope s t<2^31
  q : (kEnvelope s t/8+1)*s.n<2^31
  scale : (kEnvelope s t/s.group+1)*s.n<2^31
  zero : (kEnvelope s t/s.group+1)*(s.n/8)<2^31
  output : blocksM s t*t.bm*s.n+blocksN s t*t.bn<2^31

theorem formed_addresses (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (domain : IndexDomain s t) (r c k : Nat) (hr : r<s.m) (hc : c<s.n)
    (hk : k<kEnvelope s t) :
    aAddress s r k<2^31 ∧ qAddress s k c<2^31 ∧ sAddress s k c<2^31 ∧ zAddress s k c<2^31 := by
  have hq := Nat.div_le_div_right (show k≤kEnvelope s t by omega) (c:=8)
  have hg := Nat.div_le_div_right (show k≤kEnvelope s t by omega) (c:=s.group)
  have hn8 : c/8<s.n/8 := (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr (by simpa [Nat.div_mul_cancel hs.n8] using hc)
  refine ⟨?_,?_,?_,?_⟩
  · have h := domain.a;unfold aAddress;nlinarith
  · exact lt_trans (rowMajor_bound _ _ _ _ (by omega) hc) domain.q
  · exact lt_trans (rowMajor_bound _ _ _ _ (by omega) hc) domain.scale
  · exact lt_trans (rowMajor_bound _ _ _ _ (by omega) hn8) domain.zero

theorem formed_output (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (domain : IndexDomain s t) (l : Lane) (hl : l.Valid s t) :
    cAddress s (laneWork s t l).row (laneWork s t l).col<2^31 := by
  obtain ⟨hr,hc⟩ := GroupedGrid.bounds (blocksM s t) (blocksN s t) t.gm l.pid
    (ceil_positive _ _ hs.npos ht.bnpos) ht.gmpos hl.1
  have rr := rowMajor_bound (blocksM s t) t.bm _ l.i hr hl.2.2.1
  have cc := rowMajor_bound (blocksN s t) t.bn _ l.j hc hl.2.2.2
  have h := domain.output
  dsimp [cAddress,laneWork]
  nlinarith

theorem signed32_exact (n : Nat) (h : n<2^31) : (BitVec.ofNat 32 n).toInt=(n:Int) := by
  rw [BitVec.toInt_eq_toNat_of_lt]
  · simp
    omega
  · simp [Nat.mod_eq_of_lt (show n<2^32 by omega)];omega

theorem byte_offset_bound (n : Nat) (h : n<2^31) (i : Fin 4) : 4*n+i.val<2^63 := by omega
#axiomsClean active_addresses
#axiomsClean formed_k
#axiomsClean packed_advance
#axiomsClean formed_addresses
#axiomsClean formed_output
#axiomsClean signed32_exact
#axiomsClean byte_offset_bound
end VeriTile.Bench.Repairs.Int4Matmul
