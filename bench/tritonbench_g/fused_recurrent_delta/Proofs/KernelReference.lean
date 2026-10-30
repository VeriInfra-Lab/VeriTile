import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRegisters
/-! Pristine-input recurrences. These definitions neither execute source syntax
nor read a reached state. Padded coordinates and every scalar operand order
are retained; Real and native interpretations are separate instantiations. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReference
open VeriTile.Triton FP.Structural KernelRegisters
noncomputable section
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def scalar (M : Algebra α) (scale : ℝ) := M.literal (some .fp32) .real scale
def add (M : Algebra α) := M.binary (some .fp32) .real .add
def sub (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul (M : Algebra α) := M.binary (some .fp32) .real .mul
def cast (M : Algebra α) := M.cast (some .fp32) .real .real
def stored (M : Algebra α) (value : α) : Cell α := .mk .real (cast M value)
def read (M : Algebra α) (u : State α) (r : RegionName) (offset : Nat) (active : Bool) : α :=
  cast M (if active then (u.mem r offset).read .real else zero M)

def keySum (M : Algebra α) (s : Shape) (values : ForwardTile s α) (j : Fin s.FBV) : α :=
  M.reduceSum (some .fp32) (shape:=[s.FBV,s.BK]) ⟨1,by simp⟩ Bool.false values (j,PUnit.unit)

structure ForwardInput (s : Shape) (α : Type) where
  key : Nat → Fin s.BK → α
  query : Nat → Fin s.BK → α
  value : Nat → Fin s.FBV → α
  beta : Nat → Fin s.FBV → α
  seed : ForwardTile s α

def forwardInput (M : Algebra α) (s : Shape) (f : Flags) (bh iv : Nat) (u : State α) : ForwardInput s α where
  key := fun t i => read M u "k" (keyAddress s bh t i.val) (decide (i.val<s.K))
  query := fun t i => read M u "q" (keyAddress s bh t i.val) (decide (i.val<s.K))
  value := fun t j => read M u "v" (valAddress s bh iv s.FBV t j.val) (decide (iv*s.FBV+j.val<s.V))
  beta := fun t j => if f.vectorBeta then
      read M u "beta" (valAddress s bh iv s.FBV t j.val) (decide (iv*s.FBV+j.val<s.V))
    else read M u "beta" (betaAddress s bh t) Bool.true
  seed := fun ji => if f.initial then add M (zero M)
    (read M u "h0" (stateAddress s bh iv s.FBV ji.2.1.val ji.1.val)
      (decide (ji.2.1.val<s.K) && decide (iv*s.FBV+ji.1.val<s.V))) else zero M

def forward (M : Algebra α) (s : Shape) (a : ForwardInput s α) : Nat → ForwardTile s α
  | 0 => a.seed
  | t+1 => fun ji =>
      let old := forward M s a t
      let residual := sub M (a.value t ji.1)
        (keySum M s (fun vi => mul M (old vi) (a.key t vi.2.1)) ji.1)
      add M (old ji) (mul M (a.key t ji.2.1) (mul M residual (a.beta t ji.1)))
def residual (M : Algebra α) (s : Shape) (a : ForwardInput s α) (t : Nat) (j : Fin s.FBV) :=
  sub M (a.value t j) (keySum M s (fun ji => mul M (forward M s a t ji) (a.key t ji.2.1)) j)
def output (M : Algebra α) (s : Shape) (a : ForwardInput s α) (scale : ℝ) (t : Nat) (j : Fin s.FBV) :=
  keySum M s (fun ji => mul M (forward M s a (t+1) ji) (mul M (a.query t ji.2.1) (scalar M scale))) j

#specNonCircular forward avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
#specNonCircular residual avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
#specNonCircular output avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReference
