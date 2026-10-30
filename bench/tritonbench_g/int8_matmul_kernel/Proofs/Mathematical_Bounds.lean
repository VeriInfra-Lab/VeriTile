import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Host
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- Supported signed32 index domain, not source runtime checks. Input bounds
include all padded initial lanes and the final unused pointer increments.
Offsets a0/b0 are in the tensor data pointer and are handled at 64-bit below. -/
structure SourceBounds (s : Shape) (b : Buffers) : Prop where
  shape : max s.m (max s.n s.k)+512<2^31
  strides : max b.am (max b.ak (max b.bk b.bn))+256<2^31
  grid : ∀ c : Fin 15,s.programs c<2^31 ∧ (configs c).gm*s.cols c<2^31
  aExpr : s.m*b.am+(s.k+128)*b.ak<2^31
  bExpr : (s.k/4+128)*b.bk+s.n*b.bn<2^31
  outExpr : (s.m+256)*max s.n 1+s.n+256<2^31

theorem formed_output (s : Shape) (b : Buffers) (c : Fin 15) (safe : SourceBounds s b)
    (p i j : Nat) (hp : p<s.programs c) (hi : i<(configs c).bm) (hj : j<(configs c).bn) :
    row s c p i<2^31 ∧ col s c p j<2^31 ∧
    row s c p i*max s.n 1+col s c p j<2^31 := by
  have sizes := config_bounds c
  have bm : 0<(configs c).bm := by omega
  have bn : 0<(configs c).bn := by omega
  have g := grouped_bounds (s.rows c) (s.cols c) (configs c).gm p (by omega) hp
  have hr := tile_extent s.m (configs c).bm _ i bm g.2.1 hi
  have hc := tile_extent s.n (configs c).bn _ j bn g.2.2 hj
  change row s c p i<s.m+(configs c).bm at hr
  change col s c p j<s.n+(configs c).bn at hc
  have sm : s.m≤max s.m (max s.n s.k) := le_max_left _ _
  have sn : s.n≤max s.m (max s.n s.k) := le_trans (le_max_left _ _) (le_max_right _ _)
  have product := Nat.mul_le_mul_right (max s.n 1) (show row s c p i≤s.m+256 by omega)
  have shape := safe.shape
  have out := safe.outExpr
  exact ⟨by omega,by omega,by omega⟩

theorem formed_a (s : Shape) (b : Buffers) (c : Fin 15) (safe : SourceBounds s b)
    (legal : s.legal c) (m advance lane : Nat) (hm : m<s.m)
    (ha : advance≤4*s.steps c) (hl : lane<(configs c).bk) :
    m*b.am+(advance*(configs c).bk+lane)*b.ak<2^31 := by
  have hk := (legal_decomposition s c legal).2
  have sizes := config_bounds c
  have hmul := Nat.mul_le_mul_right (configs c).bk ha
  have pos : advance*(configs c).bk+lane≤s.k+128 := by nlinarith
  have h1 := Nat.mul_le_mul_right b.am (Nat.le_of_lt hm)
  have h2 := Nat.mul_le_mul_right b.ak pos
  have bound := safe.aExpr
  omega

theorem formed_b (s : Shape) (b : Buffers) (c : Fin 15) (safe : SourceBounds s b)
    (legal : s.legal c) (n advance lane : Nat) (hn : n<s.n)
    (ha : advance≤s.steps c) (hl : lane<(configs c).bk) :
    (advance*(configs c).bk+lane)*b.bk+n*b.bn<2^31 := by
  have hk := (legal_decomposition s c legal).1
  have sizes := config_bounds c
  have hmul := Nat.mul_le_mul_right (configs c).bk ha
  have pos : advance*(configs c).bk+lane≤s.k/4+128 := by nlinarith
  have h1 := Nat.mul_le_mul_right b.bk pos
  have h2 := Nat.mul_le_mul_right b.bn (Nat.le_of_lt hn)
  have bound := safe.bExpr
  omega

/-- Address binding supplies actual allocation base byte addresses. This is a
pointer arithmetic bound, not an assertion that typed observations constitute
an independently verified physical CUDA memory execution. -/
structure PointerBounds (s : Shape) (b : Buffers) where
  aBase : Nat
  bBase : Nat
  cBase : Nat
  a64 : aBase+4*(b.a0+(s.m*b.am+(s.k+128)*b.ak))<2^63
  b64 : bBase+(b.b0+((s.k/4+128)*b.bk+s.n*b.bn))<2^63
  c64 : cBase+4*((s.m+256)*max s.n 1+s.n+256)<2^63

theorem a_byte_bound (s : Shape) (b : Buffers) (bounds : PointerBounds s b) (offset : Nat)
    (h : offset≤s.m*b.am+(s.k+128)*b.ak) :
    bounds.aBase+4*(b.a0+offset)<2^63 := by have := bounds.a64; omega

theorem b_byte_bound (s : Shape) (b : Buffers) (bounds : PointerBounds s b) (offset : Nat)
    (h : offset≤(s.k/4+128)*b.bk+s.n*b.bn) :
    bounds.bBase+(b.b0+offset)<2^63 := by have := bounds.b64; omega

theorem c_byte_bound (s : Shape) (b : Buffers) (bounds : PointerBounds s b) (offset : Nat)
    (h : offset≤(s.m+256)*max s.n 1+s.n+256) :
    bounds.cBase+4*offset<2^63 := by have := bounds.c64; omega

/-- Active source operands are covered by the original tensor allocations;
formed padded/final pointers require the bounds above but are not loaded. -/
theorem input_allocation_coverage (r : Request) (out : Nat) (domain : r.Domain)
    (m k q n : Nat) (hm : m<r.shape.m) (hk : k<r.shape.k)
    (hq : q<r.shape.k/4) (hn : n<r.shape.n) :
    (aAddress (hostBuffers r out) m k).2<r.a.capacity ∧
    (bAddress (hostBuffers r out) q n).2<r.b.capacity := by
  have aCovered := domain.2.2.2.2.2.2.2.2.2.2.1
  have bCovered := domain.2.2.2.2.2.2.2.2.2.2.2
  have packed : r.shape.k/4=r.b.rows := by
    change r.a.cols/4=r.b.rows
    rw [domain.1]
    omega
  exact ⟨aCovered m k hm hk,bCovered q n (by simpa only [packed] using hq) hn⟩

#axiomsClean formed_output
#axiomsClean formed_a
#axiomsClean formed_b
#axiomsClean a_byte_bound
#axiomsClean b_byte_bound
#axiomsClean c_byte_bound
#axiomsClean input_allocation_coverage
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
