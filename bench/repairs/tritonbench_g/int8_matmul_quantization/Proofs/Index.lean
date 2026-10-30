import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Source

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000
abbrev RuntimeIndex := Int8DequantMatmul.RuntimeIndex
abbrev addressWord64 := Int8DequantMatmul.addressWord64
abbrev padM (s : Shape) (c : Config) := Int8DequantMatmul.padM s c.toConfig
abbrev padN (s : Shape) (c : Config) := Int8DequantMatmul.padN s c.toConfig
abbrev formedK (s : Shape) (c : Config) := Int8DequantMatmul.formedK s c.toConfig
structure FormedRectangle where
  offset : Nat
  rows : Nat
  cols : Nat
  rowStride : Nat
  colStride : Nat
  bytes : Nat
def FormedRectangle.Safe (p : FormedRectangle) : Prop :=
  0<p.bytes ∧ p.bytes*(p.offset+p.rows*p.rowStride+p.cols*p.colStride)<2^63
theorem FormedRectangle.exact (p : FormedRectangle) (safe : p.Safe) (r k : Nat)
    (hr : r≤p.rows) (hk : k≤p.cols) :
    (addressWord64 p.offset r k p.rowStride p.colStride).toInt=
      (p.offset+r*p.rowStride+k*p.colStride : Nat) ∧
    p.bytes*(p.offset+r*p.rowStride+k*p.colStride)<2^63 := by
  have bound := Nat.add_le_add (Nat.add_le_add_left (Nat.mul_le_mul_right p.rowStride hr) p.offset)
    (Nat.mul_le_mul_right p.colStride hk)
  have byteBound := (Nat.mul_le_mul_left p.bytes bound).trans_lt safe.2
  exact ⟨Int8DequantMatmul.addressWord64_exact _ _ _ _ _ (by nlinarith [safe.1]),byteBound⟩
def aRectangle (s : Shape) (c : Config) (b : Buffers) : FormedRectangle :=
  ⟨b.aOffset,s.M,formedK s c,max s.K 1,1,1⟩
def bRectangle (s : Shape) (c : Config) (b : Buffers) : FormedRectangle :=
  ⟨b.bOffset,formedK s c,s.N,if b.bColumnMajor then 1 else max s.N 1,
    if b.bColumnMajor then max s.K 1 else 1,1⟩
def outRectangle (s : Shape) (c : Config) (b : Buffers) : FormedRectangle :=
  ⟨b.outOffset,padM s c,padN s c,max s.N 1,1,if b.outPrecision=.half then 2 else 4⟩
structure IndexABI (s : Shape) (c : Config) (b : Buffers) : Prop where
  dimensions : s.M+c.BM<2^63 ∧ s.N+c.BN<2^63 ∧ s.K+c.BK*c.S<2^63
  coordinates : padM s c<2^63 ∧ padN s c<2^63 ∧ formedK s c<2^63
  groupedWidth : c.group*((s.N+c.BN-1)/c.BN)<2^63
  gridX : ((s.M+c.BM-1)/c.BM)*((s.N+c.BN-1)/c.BN)≤2^31-1
  gridY : c.S≤65535
  aFormed : (aRectangle s c b).Safe
  bFormed : (bRectangle s c b).Safe
  outFormed : (outRectangle s c b).Safe
  aScaleFormed : 4*(b.aScaleOffset+s.M)<2^63
  bScaleFormed : 4*(b.bScaleOffset+s.N)<2^63
theorem sourceK_formed (s : Shape) (c : Config) (t z l : Nat)
    (ht : t≤steps s c) (hz : z<c.S) (hl : l<c.BK) : sourceK c t z l<formedK s c :=
  Int8DequantMatmul.formedK_bound s c.toConfig t z l ht hz hl
theorem IndexABI.a_exact (abi : IndexABI s c b) (r t z l : Nat)
    (hr : r<s.M) (ht : t≤steps s c) (hz : z<c.S) (hl : l<c.BK) :
    (addressWord64 b.aOffset r (sourceK c t z l) (max s.K 1) 1).toInt=
      (b.aOffset+r*(max s.K 1)+sourceK c t z l : Nat) := by
  simpa [aRectangle] using ((aRectangle s c b).exact abi.aFormed r (sourceK c t z l)
    hr.le (sourceK_formed s c t z l ht hz hl).le).1
structure QuantIndexABI (G K BK : Nat) (q : QuantBuffers) : Prop where
  blockPositive : 0<BK
  covers : K≤BK
  rowDomain : G=0 ∨ 0<K
  gridX : G≤2^31-1
  dimensions : G<2^63 ∧ K+BK<2^63
  inputFormed : (FormedRectangle.mk q.inputOffset G (2*BK)
    q.inputGroupStride q.inputElementStride 4).Safe
  outputFormed : 1*(q.outputOffset+G*K+2*BK)<2^63
  scalesFormed : 4*(q.scaleOffset+G)<2^63
theorem positive_width_one_iteration (K BK : Nat) (positive : 0<K) (bk : 0<BK) (cover : K≤BK) :
    (K+BK-1)/BK=1 := by
  apply Nat.div_eq_of_lt_le <;> omega
/-- Concrete ordinary contiguous FP32 row index, without allocating a tensor. -/
theorem original_signed32_row_wrap :
    ((BitVec.ofNat 32 65536)*(BitVec.ofNat 32 32768)).toInt= -2147483648 := by decide
theorem repaired_signed64_row_exact :
    ((BitVec.ofNat 64 65536)*(BitVec.ofNat 64 32768)).toInt=2147483648 := by decide
#axiomsClean FormedRectangle.exact
#axiomsClean IndexABI.a_exact
#axiomsClean positive_width_one_iteration
#axiomsClean original_signed32_row_wrap
#axiomsClean repaired_signed64_row_exact
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
