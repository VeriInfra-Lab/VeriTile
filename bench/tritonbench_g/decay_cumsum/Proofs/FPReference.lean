import bench.tritonbench_g.decay_cumsum.Proofs.FPProgram

/-! Independent scalar recurrences over logical input payloads. No interpreter
result appears in these definitions. Direct FP32 loads retain fp32Load;
explicit gate promotions retain cast; neither is assumed to be an identity.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.FPReference
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type}

def zero (M : Algebra α) : α := M.literal (some .fp32) .real 0
def add (M : Algebra α) : α → α → α := M.binary (some .fp32) .real .add
def sub (M : Algebra α) : α → α → α := M.binary (some .fp32) .real .sub
def mul (M : Algebra α) : α → α → α := M.binary (some .fp32) .real .mul
def exp2 (M : Algebra α) : α → α := M.unary (some .fp32) .exp2
def gate (M : Algebra α) : α → α := M.cast (some .fp32) .real .real
def stored (M : Algebra α) (value : α) : Cell α := .mk .real (M.cast (some .fp32) .real .real value)

def forwardCarry (M : Algebra α) (g : Nat → α) : Nat → α
  | 0 => zero M
  | n+1 => add M (forwardCarry M g n)
      (mul M (gate M (g n)) (M.literal (some .fp32) .real 1.44269504))

def forward (M : Algebra α) (g : Nat → α) (t : Nat) : Cell α :=
  stored M (forwardCarry M g (t+1))

def prepareQ (M : Algebra α) (scale : ℝ) (q g : α) : α :=
  mul M (M.fp32Load q) (mul M (exp2 M (gate M g)) (M.literal (some .fp32) .real scale))

def prepareK (M : Algebra α) (k last g : α) : α :=
  mul M (M.fp32Load k) (exp2 M (sub M (M.fp32Load last) (gate M g)))

def backwardQ (M : Algebra α) (dqi dqo g : α) : α :=
  add M (M.fp32Load dqi) (mul M (M.fp32Load dqo) (exp2 M (gate M g)))

def backwardK (M : Algebra α) (dki dko last g : α) : α :=
  add M (M.fp32Load dki) (mul M (M.fp32Load dko) (exp2 M (sub M (gate M last) (gate M g))))

def gradient (M : Algebra α) (dq dk q k : α) : α :=
  sub M (mul M dq (M.fp32Load q)) (mul M dk (M.fp32Load k))

def backwardCarry (M : Algebra α) (span : Nat)
    (dqi dqo dki dko q k g : Nat → α) : Nat → α
  | 0 => zero M
  | n+1 =>
      let t := span-1-n
      add M (backwardCarry M span dqi dqo dki dko q k g n)
        (gradient M (backwardQ M (dqi t) (dqo t) (g t))
          (backwardK M (dki t) (dko t) (g (span-1)) (g t)) (q t) (k t))

def backwardG (M : Algebra α) (span : Nat)
    (dqi dqo dki dko q k g : Nat → α) (t : Nat) : Cell α :=
  stored M (backwardCarry M span dqi dqo dki dko q k g (span-t))

end
end VeriTile.Bench.Repairs.DecayCumsum.FPReference
