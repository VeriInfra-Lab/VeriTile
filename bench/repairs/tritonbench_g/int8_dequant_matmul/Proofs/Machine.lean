import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Grid
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Machine

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

abbrev Dense2 := DequantizeMatmul.Dense2
def padM (s : Shape) (c : Config) := ((s.M+c.BM-1)/c.BM)*c.BM
def padN (s : Shape) (c : Config) := ((s.N+c.BN-1)/c.BN)*c.BN
/-- Includes the post-loop A/B pointer increment, even if never dereferenced. -/
def formedK (s : Shape) (c : Config) := (steps s c+1)*(c.BK*c.S)

theorem launched_coordinates (s : Shape) (c : Config) (pid : Nat)
    (lane : IntScaledMatmul.Revision001.Geometry.Lane (geometry s c))
    (hp : pid<IntScaledMatmul.Revision001.Geometry.count (geometry s c)) :
    IntScaledMatmul.Revision001.Geometry.outRow (geometry s c) pid lane<padM s c ∧
    IntScaledMatmul.Revision001.Geometry.outColumn (geometry s c) pid lane<padN s c := by
  let gm := (s.M+c.BM-1)/c.BM
  let gn := (s.N+c.BN-1)/c.BN
  have hgn : 0<gn := by
    change pid<gm*gn at hp
    by_contra h
    have : gn=0 := Nat.eq_zero_of_not_pos h
    simp [this] at hp
  have h := IntScaledMatmul.Revision001.GroupedGrid.bounds gm gn 8 pid hgn (by decide) hp
  have hm := Nat.mul_le_mul_right c.BM h.1
  have hn := Nat.mul_le_mul_right c.BN h.2
  simp only [Nat.succ_mul] at hm hn
  have lm := lane.1.isLt
  have ln := lane.2.1.isLt
  change lane.1.val<c.BM at lm
  change lane.2.1.val<c.BN at ln
  constructor
  · change IntScaledMatmul.Revision001.GroupedGrid.row gm gn 8 pid*c.BM+lane.1.val<gm*c.BM
    omega
  · change IntScaledMatmul.Revision001.GroupedGrid.col gm gn 8 pid*c.BN+lane.2.1.val<gn*c.BN
    omega

inductive RuntimeIndex where
  | i32 (bits : BitVec 32)
  | i64 (bits : BitVec 64)
def RuntimeIndex.decode : RuntimeIndex → Int
  | .i32 x => x.toInt
  | .i64 x => x.toInt
def RuntimeIndex.widen : RuntimeIndex → BitVec 64
  | .i32 x => x.signExtend 64
  | .i64 x => x
theorem RuntimeIndex.exact (x : RuntimeIndex) : x.widen.toInt=x.decode := by
  cases x with
  | i32 bits => exact BitVec.toInt_signExtend_of_le (by decide)
  | i64 bits => rfl

/-- Concrete, nonnumeric source-index ABI. These bounds include padded lanes,
inert singleton strides, cdiv numerators, byte scaling, storage offsets, and
the final pointer advance. They do not constrain integer dot overflow. -/
structure IndexABI (s : Shape) (c : Config) (b : Buffers)
    (da : Dense2 s.M s.K) (db : Dense2 s.K s.N) (dc : Dense2 s.M s.N) : Prop where
  dimensions : s.M+c.BM<2^63 ∧ s.N+c.BN<2^63 ∧ s.K+c.BK*c.S<2^63
  strides : da.rowStride<2^63 ∧ da.colStride<2^63 ∧ db.rowStride<2^63 ∧
    db.colStride<2^63 ∧ dc.rowStride<2^63 ∧ dc.colStride<2^63
  coordinates : padM s c<2^63 ∧ padN s c<2^63 ∧ formedK s c<2^63
  groupedWidth : 8*((s.N+c.BN-1)/c.BN)<2^63
  aFormed : b.aOffset+s.M*da.rowStride+formedK s c*da.colStride<2^63
  bFormed : b.bOffset+formedK s c*db.rowStride+s.N*db.colStride<2^63
  cBytesFormed : 2*(b.outOffset+padM s c*dc.rowStride+padN s c*dc.colStride)<2^63
  xBytesFormed : 4*(b.xOffset+s.M)<2^63
  wBytesFormed : 4*(b.wOffset+s.N)<2^63
  biasBytesFormed : 4*(b.biasOffset+padN s c)<2^63

/-- Generic CUDA grid-product constraints are separate from source expression
arithmetic. They are ordinary launch admissibility, never a numeric oracle. -/
structure GridABI (s : Shape) (c : Config) : Prop where
  x : ((s.M+c.BM-1)/c.BM)*((s.N+c.BN-1)/c.BN)≤2^31-1
  y : c.S≤65535

theorem formedK_bound (s : Shape) (c : Config) (t z l : Nat)
    (ht : t≤steps s c) (hz : z<c.S) (hl : l<c.BK) : sourceK c t z l<formedK s c := by
  have tail : z*c.BK+l<c.BK*c.S := by nlinarith
  have ht' := Nat.mul_le_mul_right (c.BK*c.S) ht
  simp only [formedK,sourceK,Nat.add_mul,Nat.one_mul]
  omega
theorem loop_start_bound (s : Shape) (c : Config) (t : Nat) (ht : t<steps s c) :
    t*(c.BK*c.S)<s.K := by
  have le := Nat.div_mul_le_self (s.K+c.BK*c.S-1) (c.BK*c.S)
  have lt := Nat.mul_le_mul_right (c.BK*c.S) ht
  change steps s c*(c.BK*c.S)≤s.K+c.BK*c.S-1 at le
  simp only [Nat.succ_mul] at lt
  have positive := Nat.mul_pos c.positiveK c.positiveS
  omega

def addressWord64 (offset r k sr sk : Nat) : BitVec 64 :=
  BitVec.ofNat 64 offset+BitVec.ofNat 64 r*BitVec.ofNat 64 sr+BitVec.ofNat 64 k*BitVec.ofNat 64 sk
theorem addressWord64_exact (offset r k sr sk : Nat) (h : offset+r*sr+k*sk<2^63) :
    (addressWord64 offset r k sr sk).toInt=(offset+r*sr+k*sk : Nat) := by
  simpa only [addressWord64,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using DequantizeMatmul.signed64_exact _ h

theorem IndexABI.formed (abi : IndexABI s c b da db dc) (nonemptyM : 0<s.M) (nonemptyN : 0<s.N)
    (r n t z l : Nat) (hr : r<padM s c) (hn : n<padN s c)
    (ht : t≤steps s c) (hz : z<c.S) (hl : l<c.BK) :
    (addressWord64 b.aOffset (r%s.M) (sourceK c t z l) da.rowStride da.colStride).toInt=
      (b.aOffset+(r%s.M)*da.rowStride+sourceK c t z l*da.colStride : Nat) ∧
    (addressWord64 b.bOffset (sourceK c t z l) (n%s.N) db.rowStride db.colStride).toInt=
      (b.bOffset+sourceK c t z l*db.rowStride+(n%s.N)*db.colStride : Nat) ∧
    (addressWord64 b.outOffset r n dc.rowStride dc.colStride).toInt=
      (b.outOffset+r*dc.rowStride+n*dc.colStride : Nat) ∧
    2*(b.outOffset+r*dc.rowStride+n*dc.colStride)<2^63 ∧
    4*(b.biasOffset+n)<2^63 ∧ 4*(b.xOffset+r%s.M)<2^63 ∧ 4*(b.wOffset+n%s.N)<2^63 := by
  have rk := formedK_bound s c t z l ht hz hl
  have rm := Nat.mod_lt r nonemptyM
  have rn := Nat.mod_lt n nonemptyN
  have ha := Nat.add_le_add (Nat.add_le_add_left (Nat.mul_le_mul_right da.rowStride rm.le) b.aOffset)
    (Nat.mul_le_mul_right da.colStride rk.le)
  have hb := Nat.add_le_add (Nat.add_le_add_left (Nat.mul_le_mul_right db.rowStride rk.le) b.bOffset)
    (Nat.mul_le_mul_right db.colStride rn.le)
  have hc := Nat.add_le_add (Nat.add_le_add_left (Nat.mul_le_mul_right dc.rowStride hr.le) b.outOffset)
    (Nat.mul_le_mul_right dc.colStride hn.le)
  have hc2 := (Nat.mul_le_mul_left 2 hc).trans_lt abi.cBytesFormed
  have hb4 := (Nat.mul_le_mul_left 4 (Nat.add_le_add_left hn.le b.biasOffset)).trans_lt abi.biasBytesFormed
  have hx4 := (Nat.mul_le_mul_left 4 (Nat.add_le_add_left rm.le b.xOffset)).trans_lt abi.xBytesFormed
  have hw4 := (Nat.mul_le_mul_left 4 (Nat.add_le_add_left rn.le b.wOffset)).trans_lt abi.wBytesFormed
  exact ⟨addressWord64_exact _ _ _ _ _ (ha.trans_lt abi.aFormed),
    addressWord64_exact _ _ _ _ _ (hb.trans_lt abi.bFormed),
    addressWord64_exact _ _ _ _ _ (by omega),hc2,hb4,hx4,hw4⟩

theorem active_dense (s : Shape) (c : Config) (da : Dense2 s.M s.K) (db : Dense2 s.K s.N)
    (dc : Dense2 s.M s.N) (r n k : Nat) (hr : r<s.M) (hn : n<s.N) (hk : k<s.K) :
    (r%s.M)*da.rowStride+k*da.colStride=r*s.K+k ∧
    k*db.rowStride+(n%s.N)*db.colStride=k*s.N+n ∧
    r*dc.rowStride+n*dc.colStride=r*s.N+n := by
  rw [Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hn]
  exact ⟨da.offset _ _ hr hk,db.offset _ _ hk hn,dc.offset _ _ hr hn⟩

theorem bias_mask_safe (s : Shape) (n : Nat) (mask : n<s.N) : n<s.N := mask

#axiomsClean RuntimeIndex.exact
#axiomsClean launched_coordinates
#axiomsClean formedK_bound
#axiomsClean loop_start_bound
#axiomsClean IndexABI.formed
#axiomsClean active_dense
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
