import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSource
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.LogicalFormula

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Geometry
open VeriTile.Triton
noncomputable section

def Supported (B : Nat) : Prop := B = 16 ∨ B = 32 ∨ B = 64
def shrink (B : Nat) : Nat := max 16 (B / 2)

/-- The wrapper may complete its first launch or its single retry. The selected
block is also the persistent global state on a normal return. Failed launches
have no successful-output conclusion. -/
inductive Completion where
  | first
  | retry
  deriving DecidableEq

def selected (initial : Nat) : Completion → Nat
  | .first => initial
  | .retry => shrink initial

theorem shrink_supported (B : Nat) (h : Supported B) : Supported (shrink B) := by
  rcases h with rfl | rfl | rfl <;> norm_num [Supported, shrink]

theorem selected_supported (initial : Nat) (h : Supported initial) (path : Completion) :
    Supported (selected initial path) := by
  cases path
  · exact h
  · exact shrink_supported initial h

theorem supported_positive (B : Nat) (h : Supported B) : 0 < B := by
  rcases h with rfl | rfl | rfl <;> norm_num [Supported, shrink]

structure Shape where
  (Z H HK N R NK D B : Nat)
  kvPositive : 0 < HK
  headDivides : H % HK = 0
  normalizerCapacity : N ≤ R
  feature : ∃ k : Nat, 4 ≤ k ∧ D = 2 ^ k
  block : Supported B

def Shape.querySize (c : Shape) := c.Z * c.H * c.N * c.D
def Shape.keySize (c : Shape) := c.Z * c.HK * c.NK * c.D
def Shape.normalizerSize (c : Shape) := c.Z * c.H * c.R
def Shape.outputSize (c : Shape) := c.Z * c.H * c.NK
def Shape.queryBase (c : Shape) (z h : Nat) := (z * c.H + h) * c.N * c.D
def Shape.keyBase (c : Shape) (z h : Nat) := (z * c.HK + h / (c.H / c.HK)) * c.NK * c.D
def Shape.normalizerBase (c : Shape) (z h : Nat) := (z * c.H + h) * c.R
def Shape.outputBase (c : Shape) (z h : Nat) := (z * c.H + h) * c.NK
def Shape.programs (c : Shape) := RealLoop.tripCount c.NK c.B

theorem digit_bound (a A b B : Nat) (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  simp only [Nat.succ_mul] at h
  omega

theorem grouped_head (c : Shape) (h : Fin c.H) : h.val / (c.H / c.HK) < c.HK := by
  have hh : 0 < c.H := Nat.zero_lt_of_lt h.isLt
  have he : c.HK * (c.H / c.HK) = c.H := by simpa [c.headDivides] using Nat.div_add_mod c.H c.HK
  have hg : 0 < c.H / c.HK := by nlinarith
  apply (Nat.div_lt_iff_lt_mul hg).2
  simpa [he] using h.isLt

theorem query_bound (c : Shape) (z : Fin c.Z) (h : Fin c.H)
    (i : Fin c.N) (d : Fin c.D) :
    c.queryBase z.val h.val + i.val * c.D + d.val < c.querySize := by
  have hzh := digit_bound z.val c.Z h.val c.H z.isLt h.isLt
  have hi := digit_bound (z.val*c.H+h.val) (c.Z*c.H) i.val c.N hzh i.isLt
  have hd := digit_bound ((z.val*c.H+h.val)*c.N+i.val) (c.Z*c.H*c.N) d.val c.D hi d.isLt
  simpa [Shape.queryBase, Shape.querySize, Nat.add_mul, Nat.add_assoc] using hd

theorem key_bound (c : Shape) (z : Fin c.Z) (h : Fin c.H)
    (j : Fin c.NK) (d : Fin c.D) :
    c.keyBase z.val h.val + d.val + j.val * c.D < c.keySize := by
  have hzh := digit_bound z.val c.Z (h.val/(c.H/c.HK)) c.HK z.isLt (grouped_head c h)
  have hj := digit_bound (z.val*c.HK+h.val/(c.H/c.HK)) (c.Z*c.HK) j.val c.NK hzh j.isLt
  have hd := digit_bound ((z.val*c.HK+h.val/(c.H/c.HK))*c.NK+j.val) (c.Z*c.HK*c.NK)
    d.val c.D hj d.isLt
  simpa [Shape.keyBase, Shape.keySize, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hd

theorem normalizer_bound (c : Shape) (z : Fin c.Z) (h : Fin c.H) (i : Fin c.N) :
    c.normalizerBase z.val h.val + i.val < c.normalizerSize :=
  digit_bound _ _ _ _ (digit_bound _ _ _ _ z.isLt h.isLt) (lt_of_lt_of_le i.isLt c.normalizerCapacity)

theorem output_bound (c : Shape) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    c.outputBase z.val h.val + j.val < c.outputSize :=
  digit_bound _ _ _ _ (digit_bound _ _ _ _ z.isLt h.isLt) j.isLt

theorem owner (c : Shape) (j : Fin c.NK) :
    j.val / c.B < c.programs ∧ j.val % c.B < c.B ∧
      j.val / c.B * c.B + j.val % c.B = j.val := by
  have hB := supported_positive c.B c.block
  have hc := (RealLoop.tripCount_bounds c.NK c.B hB).1
  refine ⟨?_, Nat.mod_lt _ hB, by simpa [Nat.mul_comm] using Nat.div_add_mod j.val c.B⟩
  apply (Nat.div_lt_iff_lt_mul hB).2
  exact lt_of_lt_of_le j.isLt hc

theorem owner_unique (B j block lane : Nat) (hB : 0 < B) (hl : lane < B)
    (he : block * B + lane = j) : block = j / B ∧ lane = j % B := by
  subst j
  constructor
  · rw [Nat.mul_comm block B, Nat.mul_add_div hB, Nat.div_eq_of_lt hl, Nat.add_zero]
  · exact (Nat.mul_add_mod_of_lt hl).symm

def parameters (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : RealSource.Parameters where
  Q := "q"
  K := "k"
  M := "m"
  Out := "out"
  queryBatch := c.H * c.N * c.D
  queryHead := c.N * c.D
  queryRow := c.D
  queryDim := 1
  keyBatch := c.HK * c.NK * c.D
  keyHead := c.NK * c.D
  keyRow := c.D
  keyDim := 1
  outputBatch := c.H * c.NK
  outputHead := c.NK
  outputInner := 1
  batches := c.Z
  heads := c.H
  kvHeads := c.HK
  queryContext := c.N
  roundedContext := c.R
  keyContext := c.NK
  offset := offset
  size := size
  scale := scale
  window := window
  complement := complement
  evenM := decide (c.N % c.B = 0)
  evenN := decide (c.NK % c.B = 0)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Geometry
