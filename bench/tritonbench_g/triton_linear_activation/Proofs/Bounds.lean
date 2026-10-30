import bench.tritonbench_g.triton_linear_activation.Proofs.Config
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

/-- The actual host guards for all source-formed offsets. Launch capacity and
resource availability are separate ordinary runtime requirements. -/
def SourceBounds (s : Shape) : Prop :=
  max s.m (max s.n s.k)+256<2^31 ∧
  (s.m+256)*max s.n 1+s.n+256<2^31 ∧
  s.m*max s.k 1+s.k+256<2^31 ∧
  s.n*max s.k 1+s.k+256<2^31

theorem formed_output (s : Shape) (c : Fin 98) (safe : SourceBounds s)
    (p i j : Nat) (hp : p<s.programs c) (hi : i<(configs c).bm) (hj : j<(configs c).bn) :
    row s c p i<2^31 ∧ col s c p j<2^31 ∧
    row s c p i*max s.n 1+col s c p j<2^31 := by
  have sizes := config_bounds c
  have bm : 0<(configs c).bm := by omega
  have bn : 0<(configs c).bn := by omega
  have g := grouped_bounds (s.rows c) (s.cols c) 8 p (by decide) hp
  have hr := tile_extent s.m (configs c).bm _ i bm g.2.1 hi
  have hc := tile_extent s.n (configs c).bn _ j bn g.2.2 hj
  change row s c p i<s.m+(configs c).bm at hr
  change col s c p j<s.n+(configs c).bn at hc
  have sm : s.m≤max s.m (max s.n s.k) := le_max_left _ _
  have sn : s.n≤max s.m (max s.n s.k) := le_trans (le_max_left _ _) (le_max_right _ _)
  have product := Nat.mul_le_mul_right (max s.n 1) (show row s c p i≤s.m+256 by omega)
  unfold SourceBounds at safe
  refine ⟨by omega,by omega,by omega⟩

/-- Covers initial padded lanes and the final pointer increment, which forms
addresses after the last dot even though those addresses are never loaded. -/
theorem formed_input (rows K block lane rowIndex steps : Nat)
    (hb : 0<block) (hb128 : block≤128) (hl : lane<block)
    (hr : rowIndex<rows) (hs : steps≤ceilDiv K block)
    (safe : rows*max K 1+K+256<2^31) :
    rowIndex*max K 1+steps*block+lane<2^31 := by
  have ht := ceil_upper K block hb
  have hm := Nat.mul_le_mul_right block hs
  have hp := Nat.mul_le_mul_right (max K 1) (Nat.le_of_lt hr)
  omega

theorem source_k_mask (K block iteration lane : Nat) (hb : 0<block)
    (ht : iteration<ceilDiv K block) (hl : lane<block) :
    (K%block=0 ∨ lane<K-iteration*block) ↔ iteration*block+lane<K := by
  by_cases exact : K%block=0
  · have he : ceilDiv K block=K/block := by
      have heq : K/block*block=K := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero exact)
      unfold ceilDiv
      have hform : K+block-1=K/block*block+(block-1) := by omega
      rw [hform,Nat.mul_comm (K/block),Nat.mul_add_div hb,
        Nat.div_eq_of_lt (show block-1<block by omega),Nat.add_zero]
    rw [he] at ht
    have heq : K/block*block=K := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero exact)
    have hmul := Nat.mul_le_mul_right block (Nat.succ_le_of_lt ht)
    simp only [exact,true_or,true_iff]
    nlinarith
  · simp only [exact,false_or]
    omega

#axiomsClean formed_output
#axiomsClean formed_input
#axiomsClean source_k_mask
end
end VeriTile.Bench.Repairs.TritonLinearActivation
