import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Host

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Index

/-- Each active full element belongs to the allocated contiguous logical slice. -/
theorem active_bound (q : Host.Request) (row k : Nat) (hr : row<q.rows) (hk : k<q.columns) :
    row*q.columns+k<q.size := by
  calc row*q.columns+k < row*q.columns+q.columns := by omega
       _=(row+1)*q.columns := by simp [Nat.add_mul]
       _≤q.rows*q.columns := Nat.mul_le_mul_right _ (by omega)

/-- Even the address expressions formed for inactive padded lanes fit signed i32. -/
theorem padded_index_bound (q : Host.Request) (row k : Nat) (hr : row<q.rows) (hk : k<16384) :
    row*q.columns+k<2^31 := by
  have hmul := Nat.mul_le_mul_right q.columns (Nat.le_of_lt hr)
  have hfits := q.indexFits
  change q.rows*q.columns+16383<2^31 at hfits
  omega

/-- Distinct row/column pairs cannot own the same output element. -/
theorem unique_owner (N row row' k k' : Nat) (hk : k<N) (hk' : k'<N)
    (he : row*N+k=row'*N+k') : row=row' ∧ k=k' := by
  have before (a b x y : Nat) (hab : a<b) (hx : x<N) : a*N+x<b*N+y := by
    calc a*N+x < a*N+N := by omega
         _=(a+1)*N := by simp [Nat.add_mul]
         _≤b*N := Nat.mul_le_mul_right _ (by omega)
         _≤b*N+y := by omega
  have hr : row=row' := by
    rcases lt_trichotomy row row' with h|h|h
    · have := before row row' k k' h hk; omega
    · exact h
    · have := before row' row k' k h hk'; omega
  exact ⟨hr,by simpa [hr] using he⟩

/-- Every valid contiguous output cell has an active owner. -/
theorem owner_exists (q : Host.Request) (o : Nat) (ho : o<q.size) :
    ∃ row k, row<q.rows ∧ k<q.columns ∧ o=row*q.columns+k := by
  refine ⟨o/q.columns,o%q.columns,?_,Nat.mod_lt _ q.positive,?_⟩
  · apply (Nat.div_lt_iff_lt_mul q.positive).2
    exact ho
  · simpa [Nat.mul_comm,Nat.add_comm] using (Nat.mod_add_div o q.columns).symm

#axiomsClean active_bound
#axiomsClean padded_index_bound
#axiomsClean unique_owner
#axiomsClean owner_exists
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Index
