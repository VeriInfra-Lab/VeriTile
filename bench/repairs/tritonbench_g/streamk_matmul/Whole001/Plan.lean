import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Foundation
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open VeriTile.Bench.TritonBenchG.StreamK
namespace Plan
set_option maxHeartbeats 4000000

/-- Original host partition, with zero K routed to the full-tile zero loop. -/
theorem edge_zero (q r : Nat) : edge q r 0=0 := by simp [edge]

theorem edge_mono (q r p p' : Nat) (h : p ≤ p') : edge q r p ≤ edge q r p' := by
  have hmul:=Nat.mul_le_mul_right q h
  have hmin:=min_le_min_right r h
  unfold edge
  omega

theorem edge_last (J P : Nat) (hP : 0 < P) : edge (J/P) (J%P) P=J := by
  have hr:=Nat.mod_lt J hP
  simp only [edge,min_eq_right (Nat.le_of_lt hr)]
  have h:=Nat.div_add_mod J P
  omega

/-- Contiguous intervals cover the full work range without a scheduler premise. -/
theorem interval_cover (q r P i : Nat) (hi : i < edge q r P) :
    ∃ p,p  <  P ∧ edge q r p  ≤  i ∧ i < edge q r (p+1) := by
  induction P with
  | zero => simp [edge] at hi
  | succ P ih =>
    by_cases h:i < edge q r P
    · obtain ⟨p,hp,hl,hr⟩:=ih h
      exact ⟨p,by omega,hl,hr⟩
    · exact ⟨P,by omega,by omega,hi⟩

/-- Every iteration has one owner, even when P exceeds the iteration count. -/
theorem interval_unique (q r p p' i : Nat)
    (hp : edge q r p  ≤  i ∧ i < edge q r (p+1))
    (hp' : edge q r p' ≤ i ∧ i < edge q r (p'+1)) : p=p' := by
  by_contra h
  rcases lt_or_gt_of_ne h with h|h
  · have hm:=edge_mono q r (p+1) p' (by omega)
    omega
  · have hm:=edge_mono q r (p'+1) p (by omega)
    omega

theorem full_partition (J P i : Nat) (hP : 0 < P) (hi : i < J) :
    ∃! p, p  <  P ∧ edge (J/P) (J%P) p  ≤  i ∧ i < edge (J/P) (J%P) (p+1) := by
  obtain ⟨p,hp,hl,hr⟩:=interval_cover (J/P) (J%P) P i (by rwa [edge_last J P hP])
  refine ⟨p,⟨hp,hl,hr⟩,?_⟩
  intro p' h'
  exact interval_unique _ _ _ _ _ ⟨h'.2.1,h'.2.2⟩ ⟨hl,hr⟩

/-- Only the final chunk of a program can be nonterminal; one scratch tile per
program therefore suffices. This is a source control property, not numerical. -/
theorem nonterminal_last (I stop current : Nat) (hI : 0 < I)
    (h : next I stop current%I≠0) : next I stop current=stop := by
  have hm:=Nat.mod_lt current hI
  have boundary : current+(I-current%I)=(current/I+1)*I := by
    have hd:=Nat.div_add_mod current I
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm (current/I) I]
    omega
  unfold next at *
  by_cases le : current+(I-current%I) ≤ stop
  · rw [min_eq_left le,boundary,Nat.mul_mod_left] at h
    contradiction
  · exact min_eq_right (by omega)

/-- The last iteration determines the unique initializing program for a tile. -/
theorem terminal_owner (S I P tile : Nat) (hI : 0 < I) (hP : 0 < P) (ht : tile < S) :
    ∃! p,p  <  P ∧ edge ((S*I)/P) ((S*I)%P) p ≤ (tile+1)*I-1 ∧
      (tile+1)*I-1 < edge ((S*I)/P) ((S*I)%P) (p+1) := by
  apply full_partition (S*I) P _ hP
  have hle:tile+1 ≤ S:=by omega
  have hm:=Nat.mul_le_mul_right I hle
  have hpos:0 < (tile+1)*I:=Nat.mul_pos (by omega) hI
  omega

#axiomsClean edge_last
#axiomsClean interval_cover
#axiomsClean interval_unique
#axiomsClean full_partition
#axiomsClean next_progress
#axiomsClean nonterminal_last
#axiomsClean terminal_owner
end Plan
end VeriTile.Bench.Repairs.StreamK.Whole001
