import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Kernels
theorem configurations_positive (c : Config) (hc : c ∈ configurations) :
    0 < c.BT ∧ c.BT ≤ 64 := by
  simp [configurations] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem source_projects (h : Call) (c : Config) :
    (source h c).toAlgorithm? = .ok (source h c).toAlgKernel := by
  cases ha : h.inputDType <;> cases hb : h.outputDType <;> rfl

#axiomsClean source_projects
end VeriTile.Bench.Repairs.ChunkCumsumVector.Kernels

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Geometry
open VeriTile.Triton

def chunks (h : Call) (c : Config) : Nat := (h.T + c.BT - 1) / c.BT
def features (h : Call) : Nat := (h.S + BS - 1) / BS

structure Program (h : Call) where
  feature : Nat
  row : Nat
  featureInside : feature < features h
  rowInside : row < h.B * h.H

def activeTime (h : Call) (c : Config) (q i : Nat) : Prop := i < h.T - q*c.BT
def activeColumn (h : Call) (feature j : Nat) : Prop := j < h.S - feature*BS
instance (h : Call) (c : Config) (q i : Nat) : Decidable (activeTime h c q i) := inferInstanceAs (Decidable (_ < _))
instance (h : Call) (feature j : Nat) : Decidable (activeColumn h feature j) := inferInstanceAs (Decidable (_ < _))

def time (h : Call) (c : Config) (q i : Nat) : Nat := q*c.BT + if activeTime h c q i then i else 0
def column (h : Call) (feature j : Nat) : Nat := feature*BS + if activeColumn h feature j then j else 0
def active (h : Call) (c : Config) (feature q i j : Nat) : Prop :=
  activeTime h c q i ∧ activeColumn h feature j
instance (h : Call) (c : Config) (feature q i j : Nat) : Decidable (active h c feature q i j) := inferInstanceAs (Decidable (_ ∧ _))

theorem ceiling_start (n block q : Nat) (positive : 0 < block)
    (inside : q < (n+block-1)/block) : q*block<n := by
  have hb := (Nat.le_div_iff_mul_le positive).mp (Nat.succ_le_of_lt inside)
  have hp : q*block+block≤n+block-1 := by simpa only [Nat.succ_mul] using hb
  omega

theorem ceiling_covers (n block : Nat) (positive : 0 < block) : n≤((n+block-1)/block)*block := by
  have hd := Nat.div_add_mod (n+block-1) block
  have hm := Nat.mod_lt (n+block-1) positive
  rw [Nat.mul_comm] at hd
  omega

theorem ceiling_le (n block : Nat) (positive : 0 < block) : (n+block-1)/block≤n := by
  apply Nat.le_of_lt_succ
  apply (Nat.div_lt_iff_lt_mul positive).mpr
  have hp : n+block-1<n+block := by omega
  have hm := Nat.mul_le_mul_left n positive
  nlinarith

theorem chunk_start (h : Call) (c : Config) (hc : 0<c.BT) (q : Nat) (hq : q<chunks h c) : q*c.BT<h.T :=
  ceiling_start h.T c.BT q hc hq

theorem feature_start (h : Call) (p : Program h) : p.feature*BS<h.S :=
  ceiling_start h.S BS p.feature (by decide) p.featureInside

theorem time_bound (h : Call) (c : Config) (hc : 0<c.BT) (q : Nat) (hq : q<chunks h c) (i : Nat) :
    time h c q i<h.T := by
  have := chunk_start h c hc q hq
  unfold time activeTime
  split <;> omega

theorem column_bound (h : Call) (p : Program h) (j : Nat) : column h p.feature j<h.S := by
  have := feature_start h p
  unfold column activeColumn
  split <;> omega

theorem active_iff (h : Call) (c : Config) (p : Program h) (q i j : Nat)
    (start : q*c.BT<h.T) :
    active h c p.feature q i j ↔ q*c.BT+i<h.T ∧ p.feature*BS+j<h.S := by
  have feature := feature_start h p
  unfold active activeTime activeColumn
  omega

theorem active_coordinates (h : Call) (c : Config) (p : Program h) (q i j : Nat)
    (ha : active h c p.feature q i j) :
    time h c q i=q*c.BT+i ∧ column h p.feature j=p.feature*BS+j := by
  exact ⟨by simp [time,ha.1],by simp [column,ha.2]⟩

theorem coordinate_cover (n block coordinate : Nat) (positive : 0<block) (inside : coordinate<n) :
    coordinate/block<(n+block-1)/block ∧ coordinate%block<block ∧
    (coordinate/block)*block+coordinate%block=coordinate := by
  have cover := ceiling_covers n block positive
  have qr := Nat.div_add_mod coordinate block
  refine ⟨?_,Nat.mod_lt _ positive,by simpa [Nat.mul_comm] using qr⟩
  apply (Nat.div_lt_iff_lt_mul positive).mpr
  omega

structure OutputLayout (h : Call) : Prop where
  distinct : h.input ≠ h.output
  unique : ∀ r s t u x y, r<h.B*h.H → s<h.B*h.H → t<h.T → u<h.T → x<h.S → y<h.S →
    h.outputStrides.address h.H r t x = h.outputStrides.address h.H s u y → r=s ∧ t=u ∧ x=y

#axiomsClean time_bound
#axiomsClean column_bound
#axiomsClean coordinate_cover
end VeriTile.Bench.Repairs.ChunkCumsumVector.Geometry
