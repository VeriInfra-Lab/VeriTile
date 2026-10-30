import bench.tritonbench_g.chunk_gate_recurrence.Proofs.Base
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section

def matrixOffset (c : Shape) (h t k v : Nat) := ((h*c.N+t)*c.K+k)*c.V+v
def seedDenseOffset (c : Shape) (h k v : Nat) := (h*c.K+k)*c.V+v
def scratchOffset (c : Shape) (h t tk tv : Nat) := (h*c.N+t)*c.tiles+tk*c.VT+tv

/-- The exact source polynomial, with its distinct tile and arange factors. -/
theorem source_address (c : Shape) (h t tk tv e j : Nat) :
    h*c.N*c.K*c.V+tk*c.V*64+e*c.V+tv*16+j+t*c.K*c.V =
      matrixOffset c h t (tk*64+e) (tv*16+j) := by
  unfold matrixOffset
  ring

theorem seed_dense_address (c : Shape) (h tk tv e j : Nat) :
    h*c.K*c.V+tk*c.V*64+e*c.V+tv*16+j = seedDenseOffset c h (tk*64+e) (tv*16+j) := by
  unfold seedDenseOffset
  ring

theorem digit_injective (n a b a' b' : Nat) (hb : b<n) (hb' : b'<n)
    (he : a*n+b=a'*n+b') : a=a' ∧ b=b' := by
  have ha : a=a' := by
    by_contra h
    rcases lt_or_gt_of_ne h with h|h
    · have hm := Nat.mul_le_mul_right n (Nat.succ_le_iff.mpr h)
      simp only [Nat.succ_mul] at hm
      omega
    · have hm := Nat.mul_le_mul_right n (Nat.succ_le_iff.mpr h)
      simp only [Nat.succ_mul] at hm
      omega
  exact ⟨ha,by simpa [ha] using he⟩

theorem matrix_injective (c : Shape) (h t k v h' t' k' v' : Nat)
    (ht : t<c.N) (hk : k<c.K) (hv : v<c.V)
    (ht' : t'<c.N) (hk' : k'<c.K) (hv' : v'<c.V)
    (he : matrixOffset c h t k v=matrixOffset c h' t' k' v') :
    h=h' ∧ t=t' ∧ k=k' ∧ v=v' := by
  obtain ⟨h1,hv1⟩ := digit_injective c.V _ _ _ _ hv hv' he
  obtain ⟨h2,hk1⟩ := digit_injective c.K _ _ _ _ hk hk' h1
  obtain ⟨hh1,ht1⟩ := digit_injective c.N _ _ _ _ ht ht' h2
  exact ⟨hh1,ht1,hk1,hv1⟩

theorem coordinate_coverage (c : Shape) (k v : Nat) (hk : k<c.K) (hv : v<c.V) :
    k/64<c.KT ∧ v/16<c.VT ∧ k%64<64 ∧ v%16<16 ∧
    k=(k/64)*64+k%64 ∧ v=(v/16)*16+v%16 ∧
    ∀ tk tv e j, tk<c.KT → tv<c.VT → e<64 → j<16 →
      k=tk*64+e → v=tv*16+j → tk=k/64 ∧ tv=v/16 ∧ e=k%64 ∧ j=v%16 := by
  dsimp [Shape.K,Shape.V] at hk hv
  refine ⟨by omega,by omega,by omega,by omega,by omega,by omega,?_⟩
  intro tk tv e j _ _ he hj hek hej
  omega

theorem main_bound (c : Shape) (h t k v : Nat) (hh : h<c.heads) (ht : t<c.N)
    (hk : k<c.K) (hv : v<c.V) : matrixOffset c h t k v<c.heads*c.N*c.K*c.V := by
  have h1 : h*c.N+t<c.heads*c.N := by
    have := Nat.mul_le_mul_right c.N (Nat.succ_le_of_lt hh)
    simp only [Nat.succ_mul] at this
    omega
  have h2 : (h*c.N+t)*c.K+k<c.heads*c.N*c.K := by
    have := Nat.mul_le_mul_right c.K (Nat.succ_le_of_lt h1)
    simp only [Nat.succ_mul] at this
    omega
  have h3 := Nat.mul_le_mul_right c.V (Nat.succ_le_of_lt h2)
  simp only [Nat.succ_mul] at h3
  dsimp [matrixOffset]
  omega

#axiomsClean source_address
#axiomsClean matrix_injective
#axiomsClean coordinate_coverage
#axiomsClean main_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
