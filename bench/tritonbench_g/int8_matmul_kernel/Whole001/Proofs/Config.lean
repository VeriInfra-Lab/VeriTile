import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Geometry
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000
structure Config where
  bm : Nat
  bn : Nat
  bk : Nat
  gm : Nat
  warps : Nat
  stages : Nat
  deriving DecidableEq
/-- All original configs in their unchanged source order. -/
def configs : Fin 15 → Config :=
  ![⟨128,256,64,8,8,3⟩,
  ⟨64,256,32,8,4,4⟩,
  ⟨128,128,32,8,4,4⟩,
  ⟨128,64,32,8,4,4⟩,
  ⟨64,128,32,8,4,4⟩,
  ⟨128,32,32,8,4,4⟩,
  ⟨128,256,128,8,8,3⟩,
  ⟨256,128,128,8,8,3⟩,
  ⟨256,64,128,8,4,4⟩,
  ⟨64,256,128,8,4,4⟩,
  ⟨128,128,128,8,4,4⟩,
  ⟨128,64,64,8,4,4⟩,
  ⟨64,128,64,8,4,4⟩,
  ⟨128,32,64,8,4,4⟩,
  ⟨32,32,32,4,4,4⟩]
theorem config_bounds (c : Fin 15) :
    32≤(configs c).bm ∧ (configs c).bm≤256 ∧
    32≤(configs c).bn ∧ (configs c).bn≤256 ∧
    32≤(configs c).bk ∧ (configs c).bk≤128 ∧
    (configs c).bk%32=0 ∧ 0<(configs c).gm ∧ (configs c).gm≤8 := by
  have all : ∀ c : Fin 15,
    32≤(configs c).bm ∧ (configs c).bm≤256 ∧
    32≤(configs c).bn ∧ (configs c).bn≤256 ∧
    32≤(configs c).bk ∧ (configs c).bk≤128 ∧
    (configs c).bk%32=0 ∧ 0<(configs c).gm ∧ (configs c).gm≤8 := by decide
  exact all c
/-- Exactly the source's positive-denominator ceil expression. -/
def ceilDiv (n block : Nat) : Nat := (n+block-1)/block

theorem ceil_cover (n block : Nat) (hb : 0<block) : n≤ceilDiv n block*block := by
  have h := Nat.mod_add_div (n+block-1) block
  have r := Nat.mod_lt (n+block-1) hb
  rw [Nat.mul_comm block] at h
  unfold ceilDiv
  omega

theorem ceil_upper (n block : Nat) (hb : 0<block) : ceilDiv n block*block<n+block := by
  have h := Nat.div_mul_le_self (n+block-1) block
  unfold ceilDiv
  omega

theorem ceil_zero (block : Nat) (hb : 0<block) : ceilDiv 0 block=0 := by
  unfold ceilDiv
  apply Nat.div_eq_of_lt
  omega

theorem ceil_positive (n block : Nat) (hn : 0<n) (hb : 0<block) : 0<ceilDiv n block := by
  have := ceil_cover n block hb
  nlinarith

theorem tile_owner_bound (n block value : Nat) (hb : 0<block) (hv : value<n) :
    value/block<ceilDiv n block := by
  apply (Nat.div_lt_iff_lt_mul hb).mpr
  exact lt_of_lt_of_le hv (ceil_cover n block hb)

theorem tile_extent (n block tile lane : Nat) (hb : 0<block)
    (ht : tile<ceilDiv n block) (hl : lane<block) : tile*block+lane<n+block := by
  have h := Nat.mul_le_mul_right block (Nat.succ_le_of_lt ht)
  have c := ceil_upper n block hb
  nlinarith

structure Shape where
  m : Nat
  n : Nat
  k : Nat
  deriving DecidableEq
def Shape.rows (s : Shape) (c : Fin 15) := ceilDiv s.m (configs c).bm
def Shape.cols (s : Shape) (c : Fin 15) := ceilDiv s.n (configs c).bn
def Shape.steps (s : Shape) (c : Fin 15) := ceilDiv (s.k/4) (configs c).bk
def Shape.programs (s : Shape) (c : Fin 15) := s.rows c*s.cols c
def Shape.legal (s : Shape) (c : Fin 15) : Prop := s.k%(4*(configs c).bk)=0
instance (s : Shape) (c : Fin 15) : Decidable (s.legal c) := inferInstanceAs (Decidable (_=0))
def row (s : Shape) (c : Fin 15) (pid lane : Nat) :=
  tileRow (s.rows c) (s.cols c) (configs c).gm pid*(configs c).bm+lane
def col (s : Shape) (c : Fin 15) (pid lane : Nat) :=
  tileCol (s.rows c) (s.cols c) (configs c).gm pid*(configs c).bn+lane
theorem source_cell_covered (s : Shape) (c : Fin 15) (m n : Nat)
    (hm : m<s.m) (hn : n<s.n) :
    ∃ p, p<s.programs c ∧ ∃ i j,
      i<(configs c).bm ∧ j<(configs c).bn ∧ row s c p i=m ∧ col s c p j=n := by
  have bm : 0<(configs c).bm := by have := config_bounds c; omega
  have bn : 0<(configs c).bn := by have := config_bounds c; omega
  obtain ⟨p,hp,hr,hc⟩ := grouped_covered (s.rows c) (s.cols c) (configs c).gm
    (m/(configs c).bm) (n/(configs c).bn) (by have := config_bounds c; omega)
    (tile_owner_bound _ _ _ bm hm) (tile_owner_bound _ _ _ bn hn)
  refine ⟨p,hp,m%(configs c).bm,n%(configs c).bn,Nat.mod_lt _ bm,Nat.mod_lt _ bn,?_,?_⟩
  · simp only [row,hr]; simpa [Nat.mul_comm] using Nat.div_add_mod m (configs c).bm
  · simp only [col,hc]; simpa [Nat.mul_comm] using Nat.div_add_mod n (configs c).bn

#axiomsClean config_bounds
#axiomsClean grouped_covered
#axiomsClean source_cell_covered
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
