import VeriTile.Triton
import VeriTile.Meta.StatementAudit
/-! Independent logical-input target, per batch/head. There is no decay,
normalization, or scale in this operation. These formulas do not refer to an
execution, a prepared carry, or a supplied gradient producer. They state the
forward map and its three vector-Jacobian products for incoming gradient `g`.
No source correctness claim is made by defining these targets. -/
namespace VeriTile.Bench.TritonBenchG.LightningAttention.LogicalTarget
noncomputable section

abbrev Matrix := Nat → Nat → ℝ

def score (d : Nat) (q k : Matrix) (t u : Nat) : ℝ :=
  ∑ a : Fin d, q t a.val * k u a.val

def output (n d : Nat) (q k v : Matrix) (t c : Nat) : ℝ :=
  ∑ u : Fin n, if u.val ≤ t then score d q k t u.val * v u.val c else 0

def queryGradient (n e : Nat) (k v g : Matrix) (t a : Nat) : ℝ :=
  ∑ u : Fin n, if u.val ≤ t then
    (∑ c : Fin e, g t c.val * v u.val c.val) * k u.val a else 0

def keyGradient (n e : Nat) (q v g : Matrix) (u a : Nat) : ℝ :=
  ∑ t : Fin n, if u ≤ t.val then
    q t.val a * (∑ c : Fin e, g t.val c.val * v u c.val) else 0

def valueGradient (n d : Nat) (q k g : Matrix) (u c : Nat) : ℝ :=
  ∑ t : Fin n, if u ≤ t.val then score d q k t.val u * g t.val c else 0

def constant (v : ℝ) : Matrix := fun _ _ => v

theorem constant_output_last (n d : Nat) (hn : 0 < n) (v : ℝ) :
    output n d (constant 1) (constant 1) (constant v) (n-1) 0 =
      (n : ℝ) * (d : ℝ) * v := by
  have hu : ∀ u : Fin n, u.val ≤ n-1 := fun u => by have := u.isLt; omega
  simp [output, score, constant, hu]
  ring

theorem constant_queryGradient_last (n e : Nat) (hn : 0 < n) (v : ℝ) :
    queryGradient n e (constant 1) (constant v) (constant 1) (n-1) 0 =
      (n : ℝ) * (e : ℝ) * v := by
  have hu : ∀ u : Fin n, u.val ≤ n-1 := fun u => by have := u.isLt; omega
  simp [queryGradient, constant, hu]
  ring

theorem constant_keyGradient_first (n e : Nat) (v : ℝ) :
    keyGradient n e (constant 1) (constant v) (constant 1) 0 0 =
      (n : ℝ) * (e : ℝ) * v := by
  simp [keyGradient, constant]
  ring

theorem constant_valueGradient_first (n d : Nat) :
    valueGradient n d (constant 1) (constant 1) (constant 1) 0 0 =
      (n : ℝ) * (d : ℝ) := by
  simp [valueGradient, score, constant]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.LightningAttention.LogicalTarget
