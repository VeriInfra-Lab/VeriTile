import bench.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Mathematical_Formula

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Kernels
theorem configurations_positive (c : Config) (hc : c ∈ configurations) :
    0<c.BT ∧ c.BT≤64 := by
  simp [configurations] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl <;> decide

theorem source_projects (h : Call) (c : Config) :
    (source h c).toAlgorithm? = .ok (source h c).toAlgKernel := by
  cases ha : h.inputDType <;> cases hb : h.outputDType <;> rfl

#axiomsClean source_projects
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Kernels

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Geometry
open VeriTile.Triton
noncomputable section

abbrev chunks (h : Call) (c : Config) := (h.T+c.BT-1)/c.BT

abbrev active (h : Call) (c : Config) (q : Nat) (i : Fin c.BT) : Prop := i.val < h.T-q*c.BT

def time (h : Call) (c : Config) (q : Nat) (i : Fin c.BT) :=
  q*c.BT + if active h c q i then i.val else 0

theorem chunk_start (h : Call) (c : Config) (positive : 0<c.BT) (q : Nat)
    (inside : q<chunks h c) : q*c.BT<h.T := by
  have hb := (Nat.le_div_iff_mul_le positive).mp (Nat.succ_le_of_lt inside)
  have hp : q*c.BT+c.BT ≤ h.T+c.BT-1 := by simpa only [Nat.succ_mul] using hb
  omega

theorem time_bound (h : Call) (c : Config) (positive : 0<c.BT) (q : Nat)
    (inside : q<chunks h c) (i : Fin c.BT) : time h c q i<h.T := by
  have := chunk_start h c positive q inside
  dsimp only [time,active]
  split <;> omega

theorem active_iff (h : Call) (c : Config) (q : Nat) (i : Fin c.BT)
    (start : q*c.BT<h.T) : active h c q i ↔ q*c.BT+i.val<h.T := by
  dsimp only [active]
  omega

theorem active_time (h : Call) (c : Config) (q : Nat) (i : Fin c.BT)
    (ha : active h c q i) : time h c q i=q*c.BT+i.val := by
  simp only [time,if_pos ha]

theorem chunks_cover (h : Call) (c : Config) (positive : 0<c.BT) : h.T≤chunks h c*c.BT := by
  have hd := Nat.div_add_mod (h.T+c.BT-1) c.BT
  have hm := Nat.mod_lt (h.T+c.BT-1) positive
  dsimp only [chunks]
  rw [Nat.mul_comm] at hd
  omega

theorem chunks_le_time (h : Call) (c : Config) (positive : 0<c.BT) : chunks h c≤h.T := by
  have hb : h.T+c.BT-1 < (h.T+1)*c.BT := by
    have := Nat.mul_le_mul_left h.T positive
    have hp : h.T+c.BT-1<h.T+c.BT := by omega
    nlinarith
  exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul positive).mpr hb)

structure OutputLayout (h : Call) : Prop where
  distinct : h.input≠h.output
  unique : ∀ r s t u, r<h.B*h.H → s<h.B*h.H → t<h.T → u<h.T →
    h.outputStrides.address h.H r t=h.outputStrides.address h.H s u → r=s ∧ t=u

theorem same_row (h : Call) (layout : OutputLayout h) (row : Nat) (hr : row<h.B*h.H)
    (t u : Nat) (ht : t<h.T) (hu : u<h.T)
    (eq : h.outputStrides.address h.H row t=h.outputStrides.address h.H row u) : t=u :=
  (layout.unique row row t u hr hr ht hu eq).2

#print axioms time_bound
#print axioms chunks_le_time
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Geometry
