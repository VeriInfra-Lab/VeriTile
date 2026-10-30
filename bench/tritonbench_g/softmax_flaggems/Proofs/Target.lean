import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealMath
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Target
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Mathematical -infinity has zero mass. No source graph occurs here. -/
def mass (q : WithBot ℝ) : ℝ := q.elim 0 Real.exp

def denominator {N : Nat} (q : Fin N → WithBot ℝ) : ℝ := ∑ n, mass (q n)
def normalized {N : Nat} (q : Fin N → WithBot ℝ) (n : Fin N) : ℝ :=
  mass (q n) / denominator q

def Defined {N : Nat} (q : Fin N → WithBot ℝ) : Prop := ∃ n, q n ≠ ⊥

def vjp {N : Nat} (p g : Fin N → ℝ) (n : Fin N) : ℝ :=
  p n * (g n - ∑ j, p j * g j)

def address (N K row col n : Nat) : Nat := row*N*K+n*K+col

def column (q : Nat → WithBot ℝ) (N K row col : Nat) (n : Fin N) : WithBot ℝ :=
  q (address N K row col n.val)

def forward (q : Nat → WithBot ℝ) (N K row col : Nat) (n : Fin N) : ℝ :=
  normalized (column q N K row col) n

def backward (q : Nat → WithBot ℝ) (g : Nat → ℝ) (N K row col : Nat) (n : Fin N) : ℝ :=
  vjp (forward q N K row col) (fun j => g (address N K row col j.val)) n

/-- Partial mathematical interpretation of a raw caller word. Negative
infinity is accepted explicitly; NaNs and positive infinity remain outside
this mathematical interpretation while the wrapper still admits those words. -/
def decodeWord (w : BitVec 32) : Option (WithBot ℝ) :=
  if w=0xff800000 then some ⊥ else
    match (Float32Bits.mk w).decodeRat with
    | none => none
    | some x => some ((x : ℝ) : WithBot ℝ)

def MathematicalWords (count : Nat) (words : Nat → BitVec 32) (q : Nat → WithBot ℝ) : Prop :=
  ∀ o, o<count → decodeWord (words o)=some (q o)

/-- Caller-coordinate extension used only at valid output addresses. -/
def flatForward (q : Nat → WithBot ℝ) (N K o : Nat) : ℝ :=
  Target.mass (q o) / denominator (column q N K (o/K/N) (o%K))

def flatBackward (p g : Nat → ℝ) (N K o : Nat) : ℝ :=
  p o * (g o - ∑ n : Fin N, p (address N K (o/K/N) (o%K) n.val) *
    g (address N K (o/K/N) (o%K) n.val))

theorem coordinates (N K row col n : Nat) (hN : 0<N) (hK : 0<K)
    (hn : n<N) (hk : col<K) :
    address N K row col n/K/N=row ∧ address N K row col n%K=col := by
  have form : address N K row col n=(row*N+n)*K+col := by
    simp [address,Nat.add_mul,Nat.mul_assoc]
  rw [form]
  constructor
  · rw [Nat.mul_comm (row*N+n) K,Nat.mul_add_div hK,
      Nat.div_eq_of_lt hk,Nat.add_zero,Nat.mul_comm row N,
      Nat.mul_add_div hN,Nat.div_eq_of_lt hn,Nat.add_zero]
  · simp [Nat.add_mod,Nat.mod_eq_of_lt hk]

theorem flatForward_coordinate (q : Nat → WithBot ℝ) (N K row col : Nat) (n : Fin N)
    (hK : 0<K) (hk : col<K) :
    flatForward q N K (address N K row col n.val)=forward q N K row col n := by
  obtain ⟨hr,hc⟩ := coordinates N K row col n.val (by have := n.isLt; omega) hK n.isLt hk
  simp only [flatForward,hr,hc,forward,normalized,column]

@[simp] theorem mass_bot : mass ⊥ = 0 := rfl
@[simp] theorem mass_coe (x : ℝ) : mass (x : WithBot ℝ) = Real.exp x := rfl

theorem mass_nonneg (q : WithBot ℝ) : 0 ≤ mass q := by
  cases q with
  | bot => exact le_rfl
  | coe x => exact (Real.exp_pos x).le

theorem denominator_pos {N : Nat} (q : Fin N → WithBot ℝ) (h : Defined q) :
    0 < denominator q := by
  obtain ⟨n,hn⟩ := h
  apply Finset.sum_pos' (fun j _ => mass_nonneg (q j))
  refine ⟨n,Finset.mem_univ _,?_⟩
  cases hq : q n with
  | bot => exact (hn hq).elim
  | coe x => exact Real.exp_pos x

theorem sum_one {N : Nat} (q : Fin N → WithBot ℝ) (h : Defined q) :
    ∑ n, normalized q n = 1 := by
  simp only [normalized,←Finset.sum_div]
  exact div_self (ne_of_gt (denominator_pos q h))

theorem shifted_mass (q : WithBot ℝ) (m : ℝ) :
    (WithBot.realExp (WithBot.realSub q (m : WithBot ℝ))).unbotD 0 =
      Real.exp (-m) * mass q := by
  cases q with
  | bot => change (0 : ℝ) = Real.exp (-m) * 0; simp
  | coe x =>
    change Real.exp (x-m)=Real.exp (-m)*Real.exp x
    rw [←Real.exp_add]
    congr 1
    ring

theorem shifted_normalized {N : Nat} (q : Fin N → WithBot ℝ) (m : ℝ) (n : Fin N) :
    (WithBot.realExp (WithBot.realSub (q n) (m : WithBot ℝ))).unbotD 0 /
      (Real.exp (-m) * denominator q) = normalized q n := by
  rw [shifted_mass]
  exact mul_div_mul_left _ _ (Real.exp_ne_zero _)

#axiomsClean denominator_pos
#axiomsClean sum_one
#axiomsClean shifted_normalized
#specNonCircular normalized avoiding [VeriTile.Triton.FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Target
