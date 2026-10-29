import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Config
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

theorem ceil_multiple (n b : Nat) (hb : 0<b) : ceilDiv (b*n) b=n := by
  unfold ceilDiv
  have h : b*n+b-1=(b-1)+b*n := by omega
  rw [h,Nat.add_mul_div_left _ _ hb,Nat.div_eq_of_lt (by omega)]
  omega

theorem legal_decomposition (s : Shape) (c : Fin 15) (legal : s.legal c) :
    s.k/4=(configs c).bk*s.steps c ∧ s.k=4*((configs c).bk*s.steps c) := by
  have hb : 0<(configs c).bk := by have := config_bounds c; omega
  have hd : 4*(configs c).bk ∣ s.k := Nat.dvd_of_mod_eq_zero legal
  obtain ⟨q,hq⟩ := hd
  have four : s.k/4=(configs c).bk*q := by
    rw [hq,Nat.mul_assoc,Nat.mul_div_right _ (by decide)]
  have steps : s.steps c=q := by
    unfold Shape.steps
    rw [four,ceil_multiple _ _ hb]
  rw [steps]
  exact ⟨four,by rw [hq]; simp only [Nat.mul_assoc]⟩

theorem legal_all_masks (s : Shape) (c : Fin 15) (legal : s.legal c)
    (f j lane : Nat) (hf : f<4) (hj : j<s.steps c) (hl : lane<(configs c).bk) :
    lane<s.k-(f*s.steps c+j)*(configs c).bk ∧ lane<s.k ∧
    (f*s.steps c+j)*(configs c).bk+lane<s.k ∧ j*(configs c).bk+lane<s.k/4 := by
  have h := legal_decomposition s c legal
  have hjm := Nat.mul_le_mul_right (configs c).bk (Nat.succ_le_of_lt hj)
  have hfm := Nat.mul_le_mul_right (s.steps c) (Nat.succ_le_of_lt hf)
  have hfield : (f*s.steps c+j+1)≤4*s.steps c := by nlinarith
  have hfull := Nat.mul_le_mul_right (configs c).bk hfield
  have ha : (f*s.steps c+j)*(configs c).bk+lane<s.k := by nlinarith [h.2]
  have hb : j*(configs c).bk+lane<s.k/4 := by nlinarith [h.1]
  exact ⟨by omega,by omega,ha,hb⟩

#axiomsClean legal_decomposition
#axiomsClean legal_all_masks
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
