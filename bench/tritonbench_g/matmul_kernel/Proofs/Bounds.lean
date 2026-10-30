import bench.tritonbench_g.matmul_kernel.Proofs.Source
namespace VeriTile.Bench.Repairs.MatmulKernel
set_option maxHeartbeats 3000000
/-- Exact scalar guards, including padded coordinates and the pointer update
after the final K iteration. No generic grid-product assertion is made. -/
structure Signed32 (c : Config) : Prop where
  dims : c.M<2^31 ∧ c.N<2^31 ∧ c.K<2^31
  blocks : c.BM<2^31 ∧ c.BN<2^31 ∧ c.BK<2^31
  paddedM : ceil c.M c.BM*c.BM<2^31
  paddedN : ceil c.N c.BN*c.BN<2^31
  paddedK : c.steps*c.BK+c.BK<2^31
  astrides : c.A.sr<2^31 ∧ c.A.sc<2^31
  bstrides : c.B.sr<2^31 ∧ c.B.sc<2^31
  cstrides : c.C.sr<2^31 ∧ c.C.sc<2^31
  aformed : (c.M-1)*c.A.sr+(c.steps*c.BK+c.BK)*c.A.sc<2^31
  bformed : (c.steps*c.BK+c.BK)*c.B.sr+(c.N-1)*c.B.sc<2^31
  cformed : (ceil c.M c.BM*c.BM)*c.C.sr+(ceil c.N c.BN*c.BN)*c.C.sc<2^31

theorem tiled_bound (p l count b : Nat) (hp : p<count) (hl : l<b) : p*b+l<count*b := by
  have le := Nat.mul_le_mul_right b (show p+1≤count by omega)
  rw [Nat.add_mul,Nat.one_mul] at le
  omega
theorem formed_k_bound (c : Config) (q l : Nat) (hq : q≤c.steps) (hl : l<c.BK) :
    q*c.BK+l<c.steps*c.BK+c.BK := by
  have := Nat.mul_le_mul_right c.BK hq
  omega
theorem source_a_capacity (c : Config) (hc : c.Valid) (r k : Nat) (hm : 0<c.M) (hk : k<c.K) :
    c.A.addr (r%c.M) k<c.A.capacity := by
  apply view_bound c.A hc.aValid
  · simpa [hc.aShape.1] using Nat.mod_lt r hm
  · simpa [hc.aShape.2] using hk
theorem source_b_capacity (c : Config) (hc : c.Valid) (k n : Nat) (hn : 0<c.N) (hk : k<c.K) :
    c.B.addr k (n%c.N)<c.B.capacity := by
  apply view_bound c.B hc.bValid
  · simpa [hc.bShape.1] using hk
  · simpa [hc.bShape.2] using Nat.mod_lt n hn
theorem source_c_capacity (c : Config) (hc : c.Valid) (w : Work) (hw : c.Active w) :
    c.C.addr w.1 w.2<c.C.capacity := by
  apply view_bound c.C hc.cValid
  · simpa [hc.cShape.1] using hw.1
  · simpa [hc.cShape.2] using hw.2

theorem source_a_formed (c : Config) (hs : Signed32 c) (r q l : Nat)
    (hm : 0<c.M) (hq : q≤c.steps) (hl : l<c.BK) :
    c.A.addr (r%c.M) (q*c.BK+l)<2^31 := by
  have mr := Nat.mod_lt r hm
  have ar := Nat.mul_le_mul_right c.A.sr (show r%c.M≤c.M-1 by omega)
  have ak := Nat.mul_le_mul_right c.A.sc (Nat.le_of_lt (formed_k_bound c q l hq hl))
  have h := hs.aformed
  unfold View.addr
  omega
theorem source_b_formed (c : Config) (hs : Signed32 c) (q l n : Nat)
    (hn : 0<c.N) (hq : q≤c.steps) (hl : l<c.BK) :
    c.B.addr (q*c.BK+l) (n%c.N)<2^31 := by
  have mn := Nat.mod_lt n hn
  have bn := Nat.mul_le_mul_right c.B.sc (show n%c.N≤c.N-1 by omega)
  have bk := Nat.mul_le_mul_right c.B.sr (Nat.le_of_lt (formed_k_bound c q l hq hl))
  have h := hs.bformed
  unfold View.addr
  omega
theorem source_c_formed (c : Config) (hs : Signed32 c) (pid lane : Work)
    (hp : pid∈programs c) (hl : lane∈lanes c) :
    c.C.addr (workOf c pid lane).1 (workOf c pid lane).2<2^31 := by
  rcases pid with ⟨pm,pn⟩
  rcases lane with ⟨lm,ln⟩
  simp only [programs,lanes,List.mem_product,List.mem_range] at hp hl
  have mr := tiled_bound _ _ _ _ hp.1 hl.1
  have mn := tiled_bound _ _ _ _ hp.2 hl.2
  have ar := Nat.mul_le_mul_right c.C.sr (Nat.le_of_lt mr)
  have an := Nat.mul_le_mul_right c.C.sc (Nat.le_of_lt mn)
  have h := hs.cformed
  simp only [View.addr,workOf]
  omega

theorem source_pointer_increment (c : Config) (r q l : Nat) :
    c.A.addr r (q*c.BK+l)+c.BK*c.A.sc=c.A.addr r ((q+1)*c.BK+l) ∧
    c.B.addr (q*c.BK+l) r+c.BK*c.B.sr=c.B.addr ((q+1)*c.BK+l) r := by
  constructor <;> simp [View.addr] <;> ring
#axiomsClean source_a_capacity
#axiomsClean source_b_capacity
#axiomsClean source_c_capacity
#axiomsClean source_a_formed
#axiomsClean source_b_formed
#axiomsClean source_c_formed
#axiomsClean source_pointer_increment
end VeriTile.Bench.Repairs.MatmulKernel
