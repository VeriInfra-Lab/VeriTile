import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeBits

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeAddresses
open NativeDomain NativeBits
set_option maxHeartbeats 3000000

def laneBits (index : BitVec 32) (block lane : Nat) :=
  index*bits (block : Int)+bits (lane : Int)

/-- The public candidate's output-row product reordering is valid for the
actual modular integer operation as well as for the unbounded projection. -/
theorem reordered_lane (index : BitVec 32) (block lane : Nat) :
    bits (block : Int)*index+bits (lane : Int) = laneBits index block lane := by
  rw [BitVec.mul_comm]
  rfl

/-- Source start_m/start_n/offs_k multiplication and subsequent arange add.
The result bound includes every lane before any where-select or mask. -/
theorem lane_exact (index block lane : Nat) (indexBits : BitVec 32)
    (hi : Exact indexBits index) (hb : block < limit) (hl : lane < block)
    (hout : index*block+lane < limit) :
    Exact (indexBits*bits (block : Int)) (index*block) ∧
    Exact (laneBits indexBits block lane) (index*block+lane) := by
  have hm := nat_mul_exact hi (nat_exact hb) (by omega)
  exact ⟨hm,nat_add_exact hm (nat_exact (lt_trans hl hb)) hout⟩

def clampBits (value : BitVec 32) (bound : Nat) :=
  if value.slt (bits (bound : Int)) then value else bits 0

theorem clamp_exact (value bound : Nat) (valueBits : BitVec 32)
    (hv : Exact valueBits value) (hb : bound < limit) :
    Exact (clampBits valueBits bound) (clamp bound value) := by
  unfold clampBits clamp
  rw [slt_exact hv (nat_exact hb)]
  by_cases h : value < bound
  · simpa [h] using hv
  · simpa [h] using (bits_exact (z := 0) (by norm_num [Fits]))

def offsetBits (x y : BitVec 32) (sx sy : Nat) :=
  x*bits (sx : Int)+y*bits (sy : Int)

/-- The A/B source pointer expression's two i32 products and i32 sum.
The offset bound applies to inactive K-tail lanes as well. -/
theorem offset_exact (x y sx sy : Nat) (xb yb : BitVec 32)
    (hx : Exact xb x) (hy : Exact yb y) (hsx : sx < limit) (hsy : sy < limit)
    (hoff : x*sx+y*sy < limit) :
    Exact (xb*bits (sx : Int)) (x*sx) ∧
    Exact (yb*bits (sy : Int)) (y*sy) ∧
    Exact (offsetBits xb yb sx sy) (x*sx+y*sy) := by
  have hm₁ := nat_mul_exact hx (nat_exact hsx) (by omega)
  have hm₂ := nat_mul_exact hy (nat_exact hsy) (by omega)
  exact ⟨hm₁,hm₂,nat_add_exact hm₁ hm₂ hoff⟩

/-- Application to the actual clamped A row and an arbitrary evaluated K lane. -/
theorem a_exact (p : Parameters) (h : Admissible p) (r ki rl kl : Nat)
    (hr : r < rows p) (hk : ki < count p) (hrl : rl < p.BM) (hkl : kl < p.BK)
    (rb kb : BitVec 32) (hre : Exact rb r) (hke : Exact kb ki) :
    Exact (offsetBits (clampBits (laneBits rb p.BM rl) p.M) (laneBits kb p.BK kl) p.sam p.sak)
      (clamp p.M (r*p.BM+rl)*p.sam+(ki*p.BK+kl)*p.sak) := by
  have hrow := (lane_exact r p.BM rl rb hre (h.scalar _ (by simp)) hrl
    (row_lane_lt p h r rl hr hrl)).2
  have hclamp := clamp_exact _ p.M _ hrow (h.scalar _ (by simp))
  have hcol := (lane_exact ki p.BK kl kb hke (h.scalar _ (by simp)) hkl
    (k_lane_lt p h ki kl hk hkl)).2
  exact (offset_exact _ _ p.sam p.sak _ _ hclamp hcol
    (h.scalar _ (by simp)) (h.scalar _ (by simp))
    (a_offset_lt p h _ _ (clamp_lt _ _ h.m) (lane_lt hk hkl))).2.2

theorem b_exact (p : Parameters) (h : Admissible p) (c ki cl kl : Nat)
    (hc : c < columns p) (hk : ki < count p) (hcl : cl < p.BN) (hkl : kl < p.BK)
    (cb kb : BitVec 32) (hce : Exact cb c) (hke : Exact kb ki) :
    Exact (offsetBits (laneBits kb p.BK kl) (clampBits (laneBits cb p.BN cl) p.N) p.sbk p.sbn)
      ((ki*p.BK+kl)*p.sbk+clamp p.N (c*p.BN+cl)*p.sbn) := by
  have hcol := (lane_exact c p.BN cl cb hce (h.scalar _ (by simp)) hcl
    (column_lane_lt p h c cl hc hcl)).2
  have hclamp := clamp_exact _ p.N _ hcol (h.scalar _ (by simp))
  have hrow := (lane_exact ki p.BK kl kb hke (h.scalar _ (by simp)) hkl
    (k_lane_lt p h ki kl hk hkl)).2
  exact (offset_exact _ _ p.sbk p.sbn _ _ hrow hclamp
    (h.scalar _ (by simp)) (h.scalar _ (by simp))
    (b_offset_lt p h _ _ (lane_lt hk hkl) (clamp_lt _ _ h.n))).2.2

/-- C uses two separate signed-i32 products, followed by two pointer advances.
There is no invented native i32 addition of these two products. -/
theorem c_products_exact (p : Parameters) (h : Admissible p) (r c rl cl : Nat)
    (hr : r < rows p) (hc : c < columns p) (hrl : rl < p.BM) (hcl : cl < p.BN)
    (rb cb : BitVec 32) (hre : Exact rb r) (hce : Exact cb c) :
    Exact (bits (p.scm : Int)*laneBits rb p.BM rl) (p.scm*(r*p.BM+rl)) ∧
    Exact (bits (p.scn : Int)*laneBits cb p.BN cl) (p.scn*(c*p.BN+cl)) := by
  have hrow := (lane_exact r p.BM rl rb hre (h.scalar _ (by simp)) hrl
    (row_lane_lt p h r rl hr hrl)).2
  have hcol := (lane_exact c p.BN cl cb hce (h.scalar _ (by simp)) hcl
    (column_lane_lt p h c cl hc hcl)).2
  have hoff := c_offset_lt p h _ _ (lane_lt hr hrl) (lane_lt hc hcl)
  have hm : p.scm*(r*p.BM+rl) < limit := by
    rw [Nat.mul_comm]
    omega
  have hn : p.scn*(c*p.BN+cl) < limit := by
    rw [Nat.mul_comm]
    omega
  exact ⟨nat_mul_exact (nat_exact (h.scalar _ (by simp))) hrow hm,
    nat_mul_exact (nat_exact (h.scalar _ (by simp))) hcol hn⟩

def tailBits (K : Nat) (ki : BitVec 32) (BK : Nat) := bits (K : Int)-ki*bits (BK : Int)

/-- The source K-tail subtraction and signed comparison agree even for masked
lanes; no unsigned/truncated subtraction is substituted. -/
theorem tail_exact (p : Parameters) (h : Admissible p) (ki lane : Nat)
    (hk : ki < count p) (hl : lane < p.BK) (kb : BitVec 32) (hke : Exact kb ki) :
    Exact (tailBits p.K kb p.BK) ((p.K : Int)-(ki : Int)*p.BK) ∧
    (bits (lane : Int)).slt (tailBits p.K kb p.BK) = decide (ki*p.BK+lane < p.K) := by
  have hK := h.scalar p.K (by simp)
  have hBK := h.scalar p.BK (by simp)
  have hprod : ki*p.BK < limit := by have hh := k_lane_lt p h ki lane hk hl; omega
  have hm := nat_mul_exact hke (nat_exact hBK) hprod
  have hr : Fits ((p.K : Int)-((ki*p.BK : Nat) : Int)) := by
    unfold Fits limit at *
    omega
  have ht := sub_exact (nat_exact hK) hm hr
  refine ⟨ht,?_⟩
  unfold tailBits
  rw [slt_exact (nat_exact (lt_trans hl hBK)) ht]
  congr 1
  exact propext (RealMath.signed_tail_mask p.K p.BK ki lane)

theorem output_mask_exact (p : Parameters) (h : Admissible p) (r c : Nat)
    (rb cb : BitVec 32) (hr : Exact rb r) (hc : Exact cb c) :
    (rb.slt (bits (p.M : Int)) && cb.slt (bits (p.N : Int))) = decide (r<p.M ∧ c<p.N) := by
  rw [slt_exact hr (nat_exact (h.scalar _ (by simp))),
    slt_exact hc (nat_exact (h.scalar _ (by simp)))]
  apply Bool.eq_iff_iff.mpr
  simp only [Bool.and_eq_true,decide_eq_true_eq,Int.ofNat_lt]

/-- Positive i32 indices are sign-extended exactly before global-pointer GEP. -/
theorem sign_extend_nat {x : BitVec 32} {n : Nat} (hx : Exact x n) (hn : n < limit) :
    x.signExtend 64 = BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [sign_extend_exact hx]
  symm
  exact BitVec.toInt_ofInt_eq_self (w := 64) (n := (n : Int))
    (by decide) (by omega) (by unfold limit at hn; norm_num; omega)

def addressBits (base bytes : Nat) (index : BitVec 32) : BitVec 64 :=
  BitVec.ofNat 64 base + index.signExtend 64 * BitVec.ofNat 64 bytes

/-- Explicit unsigned-64 no-wrap condition for the actual base pointer plus
element-scaled offset. This proves arithmetic, not allocation/provenance. -/
theorem address_exact (base bytes n : Nat) (index : BitVec 32)
    (hx : Exact index n) (hn : n < limit) (haddr : base+n*bytes < 2^64) :
    (addressBits base bytes index).toNat = base+n*bytes := by
  unfold addressBits
  rw [sign_extend_nat hx hn,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt haddr]

def twoAddressBits (base bytes : Nat) (first second : BitVec 32) : BitVec 64 :=
  addressBits base bytes first + second.signExtend 64 * BitVec.ofNat 64 bytes

theorem two_addresses_exact (base bytes m n : Nat) (first second : BitVec 32)
    (hm : Exact first m) (hn : Exact second n) (hmb : m < limit) (hnb : n < limit)
    (haddr : base+m*bytes+n*bytes < 2^64) :
    (twoAddressBits base bytes first second).toNat = base+m*bytes+n*bytes := by
  unfold twoAddressBits addressBits
  rw [sign_extend_nat hm hmb,sign_extend_nat hn hnb,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_mul_ofNat,
    BitVec.ofNat_add_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat,Nat.mod_eq_of_lt haddr]

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeAddresses
