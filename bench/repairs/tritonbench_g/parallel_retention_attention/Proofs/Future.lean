import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KeyValueReal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

theorem reverse_blocks (g : ℝ) (x : Nat → ℝ) (M N S extra : Nat) (hNM : N≤M) :
    (∑ j∈Finset.range N,(g^S)^(N-1-j)*∑ u∈Finset.range S,g^u*x ((M-1-j)*S+u))*g^extra=
      ∑ n∈Finset.range (N*S),g^(n+extra)*x ((M-N)*S+n) := by
  have eq : (∑ j∈Finset.range N,(g^S)^(N-1-j)*∑ u∈Finset.range S,g^u*x ((M-1-j)*S+u))=
      rightScan (g^S) (fun j => ∑ u∈Finset.range S,g^u*x ((M-N)*S+j*S+u)) N := by
    rw [rightScan_sum]
    calc
      _ = ∑ j∈Finset.range N,(g^S)^(N-1-j)*∑ u∈Finset.range S,g^u*x ((M-N)*S+(N-1-j)*S+u) := by
        apply Finset.sum_congr rfl
        intro j hj
        have hj' : j<N := Finset.mem_range.mp hj
        have he : M-1-j=M-N+(N-1-j) := by omega
        rw [he,Nat.add_mul]
      _ = _ := Finset.sum_range_reflect (fun j => (g^S)^j*∑ u∈Finset.range S,g^u*x ((M-N)*S+j*S+u)) N
  rw [eq]
  exact future_blocks g x N S ((M-N)*S) extra

def futureKey (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e : Nat) : ℝ :=
  let M := blocks s.T 32
  let N := M-(tile.chunk+1)*2
  ∑ n∈Finset.range (N*32),gamma (tile.bh%s.H)^(n+(64-r))*keyProduct s x tile r e ((M-N)*32+n)

def futureValue (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v : Nat) : ℝ :=
  let M := blocks s.T 32
  let N := M-(tile.chunk+1)*2
  ∑ n∈Finset.range (N*32),gamma (tile.bh%s.H)^(n+(64-r))*valueProduct s x tile r v ((M-N)*32+n)

def diagonalKey (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e : Nat) : ℝ :=
  ∑ n∈Finset.range 64,if r≤n then gamma (tile.bh%s.H)^(n-r)*keyProduct s x tile r e (tile.chunk*64+n) else 0

def diagonalValue (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v : Nat) : ℝ :=
  ∑ n∈Finset.range 64,if r≤n then gamma (tile.bh%s.H)^(n-r)*valueProduct s x tile r v (tile.chunk*64+n) else 0

theorem future_key_scaled (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e : Nat) :
    (∑ j∈Finset.range (blocks s.T 32-(tile.chunk+1)*2),
      (gamma (tile.bh%s.H)^32)^(blocks s.T 32-(tile.chunk+1)*2-1-j)*
        ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^u*keyProduct s x tile r e ((blocks s.T 32-1-j)*32+u))*
      (gamma (tile.bh%s.H)^(64-r)*(Real.sqrt (s.DK:ℝ))⁻¹)=
      (Real.sqrt (s.DK:ℝ))⁻¹*futureKey s x tile r e := by
  rw [← mul_assoc,reverse_blocks _ _ _ _ _ _ (Nat.sub_le _ _)]
  unfold futureKey
  ring

theorem future_value_scaled (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v : Nat) :
    (∑ j∈Finset.range (blocks s.T 32-(tile.chunk+1)*2),
      (gamma (tile.bh%s.H)^32)^(blocks s.T 32-(tile.chunk+1)*2-1-j)*
        ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(64-r)*gamma (tile.bh%s.H)^u*valueProduct s x tile r v ((blocks s.T 32-1-j)*32+u))*
      (Real.sqrt (s.DK:ℝ))⁻¹=
      (Real.sqrt (s.DK:ℝ))⁻¹*futureValue s x tile r v := by
  have eq : (∑ j∈Finset.range (blocks s.T 32-(tile.chunk+1)*2),
      (gamma (tile.bh%s.H)^32)^(blocks s.T 32-(tile.chunk+1)*2-1-j)*
        ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(64-r)*gamma (tile.bh%s.H)^u*valueProduct s x tile r v ((blocks s.T 32-1-j)*32+u))=
    (∑ j∈Finset.range (blocks s.T 32-(tile.chunk+1)*2),
      (gamma (tile.bh%s.H)^32)^(blocks s.T 32-(tile.chunk+1)*2-1-j)*
        ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^u*valueProduct s x tile r v ((blocks s.T 32-1-j)*32+u))*gamma (tile.bh%s.H)^(64-r) := by
    simp only [Finset.sum_mul,Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j hj
    apply Finset.sum_congr rfl
    intro u hu
    ring
  rw [eq,reverse_blocks _ _ _ _ _ _ (Nat.sub_le _ _)]
  unfold futureValue
  ring

theorem keyValue_phase_sums (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e v : Nat) :
    keyValueLane realOps p s x tile r e v=
      ((Real.sqrt (s.DK:ℝ))⁻¹*(futureKey s x tile r e+diagonalKey s x tile r e),
       (Real.sqrt (s.DK:ℝ))⁻¹*(futureValue s x tile r v+diagonalValue s x tile r v)) := by
  unfold keyValueLane
  simp only [real_zero]
  rw [keyValue_future_sums]
  simp only [real_mul,real_scale,real_cast,decayNat_real]
  rw [future_key_scaled,future_value_scaled]
  have hd : keyValueDiagonal realOps p s x tile r e v=
      (fun j carry =>
        (carry.1+(Real.sqrt (s.DK:ℝ))⁻¹*∑ u∈Finset.range 32,if r≤j*32+u then gamma (tile.bh%s.H)^(j*32+u-r)*keyProduct s x tile r e (tile.chunk*64+(j*32+u)) else 0,
         carry.2+(Real.sqrt (s.DK:ℝ))⁻¹*∑ u∈Finset.range 32,if r≤j*32+u then gamma (tile.bh%s.H)^(j*32+u-r)*valueProduct s x tile r v (tile.chunk*64+(j*32+u)) else 0)) := by
    funext j carry
    simpa [Nat.add_assoc] using keyValue_diagonal_step p s x tile r e v j carry
  rw [hd]
  -- There are exactly two diagonal blocks; expand their source-ordered updates.
  simp only [foldN]
  apply Prod.ext <;> simp only [Prod.fst,Prod.snd]
  · unfold diagonalKey
    rw [← sum_tiles 2 32 (fun n => if r≤n then gamma (tile.bh%s.H)^(n-r)*keyProduct s x tile r e (tile.chunk*64+n) else 0)]
    simp only [Finset.sum_range_succ,Finset.sum_range_zero,zero_add,zero_mul,one_mul]
    ring
  · unfold diagonalValue
    rw [← sum_tiles 2 32 (fun n => if r≤n then gamma (tile.bh%s.H)^(n-r)*valueProduct s x tile r v (tile.chunk*64+n) else 0)]
    simp only [Finset.sum_range_succ,Finset.sum_range_zero,zero_add,zero_mul,one_mul]
    ring

#axiomsClean reverse_blocks
#axiomsClean future_key_scaled
#axiomsClean future_value_scaled
#axiomsClean keyValue_phase_sums
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
