import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Grid

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

structure Geometry where
  batches : Nat
  heads : Nat
  capacity : Nat
  dimension : Nat
  sequenceBlock : Nat
  laneBlock : Nat
  masked : Bool

def Geometry.rows (g : Geometry) := g.batches*g.heads
def Geometry.size (g : Geometry) := g.rows*g.dimension
def Geometry.midSize (g : Geometry) := g.rows*g.capacity*g.dimension
def Geometry.logSize (g : Geometry) := g.rows*g.capacity
def Geometry.Valid (g : Geometry) : Prop :=
  0<g.dimension ∧ g.dimension≤g.laneBlock ∧
  (g.masked=Bool.true ∨ g.laneBlock=g.dimension) ∧
  0<g.sequenceBlock ∧ g.sequenceBlock≤2^31-1 ∧
  g.batches≤2^31-1 ∧ g.heads≤65535 ∧ g.laneBlock≤2^30 ∧
  (∀ size∈[g.midSize,g.logSize,g.size,g.batches],size+g.laneBlock≤(2^63-1)/4)

theorem count_eq {α : Type} [Inhabited α] (s : State α) (Q l : Nat)
    (length : (s.mem "L" (s.pids 0)).read .nat=l) (positive : 0<l) :
    count s Q=(l+Q-1)/Q := by simp [count,length,show ¬l≤0 by omega]

theorem ceil_positive (l Q : Nat) (hl : 0<l) (hq : 0<Q) : 0<(l+Q-1)/Q := by
  apply Nat.div_pos <;> omega

theorem ceil_capacity (l Q S : Nat) (hq : 0<Q) (hl : l≤S*Q) : (l+Q-1)/Q≤S := by
  apply Nat.lt_succ_iff.mp
  apply (Nat.div_lt_iff_lt_mul hq).mpr
  rw [Nat.succ_mul]
  omega

theorem length_add_fits (l Q : Nat) (hl : l≤2^31-1) (hq : Q≤2^31-1) : l+Q-1<2^63 := by omega

theorem output_offset_bound (g : Geometry) (row k : Nat) (hr : row<g.rows) (hk : k<g.dimension) :
    row*g.dimension+k<g.size := by
  dsimp [Geometry.size]
  nlinarith

theorem formed_mid_bound (g : Geometry) (row j k : Nat) (hr : row<g.rows)
    (hj : j<g.capacity) (hk : k<g.laneBlock) :
    row*(g.capacity*g.dimension)+j*g.dimension+k<g.midSize+g.laneBlock := by
  dsimp [Geometry.midSize]
  have h : row*g.capacity+j<g.rows*g.capacity := by nlinarith
  have h' : (row*g.capacity+j)*g.dimension≤g.rows*g.capacity*g.dimension :=
    Nat.mul_le_mul_right _ (Nat.le_of_lt h)
  nlinarith

theorem formed_output_bound (g : Geometry) (row k : Nat) (hr : row<g.rows) (hk : k<g.laneBlock) :
    row*g.dimension+k<g.size+g.laneBlock := by
  dsimp [Geometry.size]
  have h := Nat.mul_le_mul_right g.dimension (Nat.le_of_lt hr)
  omega

theorem log_offset_bound (g : Geometry) (row j : Nat) (hr : row<g.rows) (hj : j<g.capacity) :
    row*g.capacity+j<g.logSize := by dsimp [Geometry.logSize]; nlinarith

theorem byte_address_fits (size block offset : Nat)
    (sizeBound : size+block≤(2^63-1)/4) (address : offset<size+block) : 4*offset<2^63 := by omega

#axiomsClean ceil_positive
#axiomsClean ceil_capacity
#axiomsClean formed_mid_bound
#axiomsClean byte_address_fits
end
end VeriTile.Bench.Repairs.FlashDecode2
