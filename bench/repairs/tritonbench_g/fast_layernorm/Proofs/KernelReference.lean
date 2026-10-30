import bench.repairs.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Independent ordered scalar expressions for all three source kernels.
Every load conversion, mask, scalar conversion, reduction, and store conversion
is retained. These definitions contain no execution or output-memory reader. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.KernelReference
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type}

def literal (M : Algebra α) (x : ℝ) := M.literal (some .fp32) .real x
def cast (M : Algebra α) := M.cast (some .fp32) .real .real
def add (M : Algebra α) := M.binary (some .fp32) .real .add
def sub (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul (M : Algebra α) := M.binary (some .fp32) .real .mul
def div (M : Algebra α) := M.binary (some .fp32) .real .div
def reduce (M : Algebra α) (x : Fin B → α) : α :=
  M.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false (fun i => x i.1) PUnit.unit
def input (M : Algebra α) (N : Nat) (x : Fin B → α) (j : Fin B) : α :=
  cast M (if j.val<N then x j else literal M 0)
def mean (M : Algebra α) (N : Nat) (x : Fin B → α) : α :=
  div M (reduce M (input M N x)) (literal M N)
def centered (M : Algebra α) (N : Nat) (x : Fin B → α) (j : Fin B) : α :=
  if j.val<N then sub M (input M N x j) (mean M N x) else literal M 0
def inverse (M : Algebra α) (N : Nat) (eps : ℝ) (x : Fin B → α) : α :=
  M.unary (some .fp32) .rsqrt (add M
    (div M (reduce M (fun j => mul M (centered M N x j) (centered M N x j))) (literal M N))
    (literal M eps))
def output (M : Algebra α) (N : Nat) (eps : ℝ) (hasBias : Bool)
    (x w b : Fin B → α) (j : Fin B) : Cell α :=
  .mk .real (cast M (add M (mul M (mul M (centered M N x j) (inverse M N eps x))
    (input M N w j)) (if hasBias then input M N b j else literal M 0)))
def savedMean (M : Algebra α) (N : Nat) (x : Fin B → α) : Cell α :=
  .mk .real (cast M (mean M N x))
def savedInverse (M : Algebra α) (N : Nat) (eps : ℝ) (x : Fin B → α) : Cell α :=
  .mk .real (cast M (inverse M N eps x))
def normalized (M : Algebra α) (N : Nat) (x : Fin B → α) (avg inv : α) (j : Fin B) : α :=
  mul M (sub M (input M N x j) (cast M avg)) (cast M inv)
def weighted (M : Algebra α) (N : Nat) (dy w : Fin B → α) (j : Fin B) : α :=
  mul M (input M N dy j) (input M N w j)
def product (M : Algebra α) (N : Nat) (x dy w : Fin B → α) (avg inv : α) (j : Fin B) : α :=
  if j.val<N then mul M (weighted M N dy w j) (normalized M N x avg inv j) else literal M 0
def gradient (M : Algebra α) (N : Nat) (x dy w : Fin B → α) (avg inv : α) (j : Fin B) : Cell α :=
  .mk .real (cast M (mul M
    (sub M (sub M (weighted M N dy w j) (div M (reduce M (weighted M N dy w)) (literal M N)))
      (div M (mul M (normalized M N x avg inv j) (reduce M (product M N x dy w avg inv))) (literal M N)))
    (cast M inv)))

def affineTerm (M : Algebra α) (active : Bool) (x dy avg inv : α) : α :=
  if active then mul M (cast M dy) (mul M (sub M (cast M x) (cast M avg)) (cast M inv))
  else literal M 0
def affineCarry (M : Algebra α) (active : Bool) (x dy avg inv : Nat → α) : Nat → α × α
  | 0 => (literal M 0,literal M 0)
  | n+1 => let previous := affineCarry M active x dy avg inv n
    (add M previous.1 (affineTerm M active (x n) (dy n) (avg n) (inv n)),
      add M previous.2 (cast M (if active then dy n else literal M 0)))
def affineWeight (M : Algebra α) (active : Bool) (x dy avg inv : Nat → α) (rows : Nat) : Cell α :=
  .mk .real (cast M (affineCarry M active x dy avg inv rows).1)
def affineBias (M : Algebra α) (active : Bool) (x dy avg inv : Nat → α) (rows : Nat) : Cell α :=
  .mk .real (cast M (affineCarry M active x dy avg inv rows).2)

#specNonCircular output avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular gradient avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular affineWeight avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular affineBias avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelReference
