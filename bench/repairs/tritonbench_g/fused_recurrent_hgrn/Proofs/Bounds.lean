import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Contract
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Machine

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Checked

theorem tile_bounds (BD : Nat) (accepted : tileCheck BD=Bool.true) : 0<BD ∧ BD ≤ 128 := by
  simp only [tileCheck,decide_eq_true_eq] at accepted
  rcases accepted with rfl|rfl|rfl <;> omega

theorem padded_bound (c : Shape) (BD block lane : Nat) (tile : tileCheck BD=Bool.true)
    (hb : block<(c.D+BD-1)/BD) (hl : lane<BD) : block*BD+lane<c.D+127 := by
  have bounds := tile_bounds BD tile
  have blockBound := Nat.mul_le_mul_right BD (show block+1 ≤ (c.D+BD-1)/BD by omega)
  have divBound := Nat.div_mul_le_self (c.D+BD-1) BD
  have dpos := c.dpos
  have subtract : c.D+BD-1+1=c.D+BD := by omega
  simp only [Nat.add_mul,Nat.one_mul] at blockBound
  omega

theorem shape_bounds (c : Shape) :
    c.T ≤ c.high ∧ c.D ≤ c.high ∧ c.BH ≤ c.high ∧ 128 ≤ c.high := by
  have hp := c.bhpos
  have dp := c.dpos
  have one : c.T+1 ≤ c.BH*(c.T+1)*c.D :=
    (Nat.le_mul_of_pos_left _ hp).trans (Nat.le_mul_of_pos_right _ dp)
  have bh : c.BH ≤ c.BH*(c.T+1)*c.D :=
    (Nat.le_mul_of_pos_right _ (by omega)).trans (Nat.le_mul_of_pos_right _ dp)
  unfold Shape.high
  omega

theorem forward_bounds (c : Shape) (BD block head lane step : Nat)
    (tile : tileCheck BD=Bool.true) (hb : block<(c.D+BD-1)/BD)
    (hh : head<c.BH) (hl : lane<BD) (hs : step ≤ c.T) :
    block*BD ≤ c.high ∧ block*BD+lane ≤ c.high ∧ head*c.T ≤ c.high ∧
    head*c.T*c.D ≤ c.high ∧ head*c.D ≤ c.high ∧
    head*c.T*c.D+(block*BD+lane) ≤ c.high ∧
    head*c.T*c.D+(block*BD+lane)+step*c.D ≤ c.high ∧
    head*c.D+(block*BD+lane) ≤ c.high := by
  have pad := padded_bound c BD block lane tile hb hl
  have dp := c.dpos
  have tbound := Nat.mul_le_mul_right c.T (show head+1 ≤ c.BH by omega)
  have tbound' := Nat.mul_le_mul_left c.BH (show c.T ≤ c.T+1 by omega)
  simp only [Nat.add_mul,Nat.one_mul] at tbound
  have h1 : head*c.T+step ≤ c.BH*(c.T+1) := by omega
  have h2 := Nat.mul_le_mul_right c.D h1
  have h3 : head ≤ c.BH*(c.T+1) :=
    (Nat.le_of_lt hh).trans (Nat.le_mul_of_pos_right _ (by omega))
  have h4 := Nat.mul_le_mul_right c.D h3
  have h5 := Nat.le_mul_of_pos_right (head*c.T) dp
  simp only [Nat.add_mul] at h2
  unfold Shape.high
  omega

/-- Every source arithmetic node has a nonnegative value bounded by the
checked signed32 ceiling. Natural subtractions below occur only on reached
reverse-loop paths; the previous-output expression is formed only at i>0. -/
def forwardNodes (c : Shape) (BD block head lane step : Nat) : List Nat :=
  [c.T,c.D,BD,block,head,lane,block*BD,block*BD+lane,head*c.T,head*c.T*c.D,
   head*c.D,head*c.T*c.D+(block*BD+lane),step*c.D,
   head*c.T*c.D+(block*BD+lane)+step*c.D,head*c.D+(block*BD+lane)]

theorem forward_nodes_bound (c : Shape) (BD block head lane step : Nat)
    (tile : tileCheck BD=Bool.true) (hb : block<(c.D+BD-1)/BD)
    (hh : head<c.BH) (hl : lane<BD) (hs : step ≤ c.T) :
    ∀ x∈forwardNodes c BD block head lane step,x ≤ c.high := by
  have bounds := forward_bounds c BD block head lane step tile hb hh hl hs
  have shapes := shape_bounds c
  have tiles := tile_bounds BD tile
  have blockLe := Nat.le_mul_of_pos_right block tiles.1
  intro x hx
  simp only [forwardNodes,List.mem_cons,List.not_mem_nil,or_false] at hx
  rcases hx with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> omega

theorem reverse_bounds (c : Shape) (BD block head lane i : Nat)
    (tile : tileCheck BD=Bool.true) (hb : block<(c.D+BD-1)/BD)
    (hh : head<c.BH) (hl : lane<BD) (hi : i<c.T) :
    c.T-1 ≤ c.high ∧ c.T-1-i ≤ c.high ∧ head*c.T+i ≤ c.high ∧
    (head*c.T+i)*c.D ≤ c.high ∧ (head*c.T+i)*c.D+(block*BD+lane) ≤ c.high ∧
    (head*c.T+i-1)*c.D+(block*BD+lane) ≤ c.high := by
  have f := forward_bounds c BD block head lane i tile hb hh hl (by omega)
  have shapes := shape_bounds c
  have dp := c.dpos
  have prev : (head*c.T+i-1)*c.D ≤ (head*c.T+i)*c.D := Nat.mul_le_mul_right c.D (by omega)
  have ri := Nat.le_mul_of_pos_right (head*c.T+i) dp
  have same : (head*c.T+i)*c.D+(block*BD+lane)=head*c.T*c.D+(block*BD+lane)+i*c.D := by ring
  omega

theorem checked_i32 (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (x : Nat) (bound : x ≤ c.high) :
    VeriTile.Bench.Repairs.ChunkGateRecurrence.Machine.I32 x := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq] at accepted
  have high := accepted.1.2.2.2.1
  constructor <;> omega

#axiomsClean forward_nodes_bound
#axiomsClean reverse_bounds
#axiomsClean checked_i32
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Checked
