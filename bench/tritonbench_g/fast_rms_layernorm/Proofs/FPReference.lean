import bench.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

/-! Independent source-order FP scalar graph. Loads, casts, literals, rsqrt,
opaque ordered reduction, separate arithmetic and saved statistic stores are
retained. No source interpreter result defines the reference. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPReference
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type}

def literal (M : Algebra α) (x : ℝ) := M.literal (some .fp32) .real x
def cast (M : Algebra α) := M.cast (some .fp32) .real .real
def add (M : Algebra α) := M.binary (some .fp32) .real .add
def sub (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul (M : Algebra α) := M.binary (some .fp32) .real .mul
def div (M : Algebra α) := M.binary (some .fp32) .real .div

def padded (M : Algebra α) (h : Call) (x : Fin h.block → α) (j : Fin h.block) : α :=
  if j.val<h.cols then x j else literal M 0

def input (M : Algebra α) (h : Call) (x : Fin h.block → α) (j : Fin h.block) := cast M (padded M h x j)

def reduce (M : Algebra α) (B : Nat) (x : Fin B → α) : α :=
  M.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false (fun i => x i.1) PUnit.unit

def inv (M : Algebra α) (h : Call) (x : Fin h.block → α) : α :=
  M.unary (some .fp32) .rsqrt
    (add M (div M (reduce M h.block (fun j => mul M (input M h x j) (input M h x j)))
      (literal M (h.cols:ℝ))) (literal M h.eps))

def normalized (M : Algebra α) (h : Call) (x : Fin h.block → α) (j : Fin h.block) :=
  mul M (input M h x j) (inv M h x)

def output (M : Algebra α) (h : Call) (x w : Fin h.block → α) (j : Fin h.block) : Cell α :=
  .mk .real (cast M (if h.gemma then
    mul M (normalized M h x j) (add M (input M h w j) (literal M 1))
  else mul M (cast M (normalized M h x j)) (M.fp32Load (padded M h w j))))

def savedInv (M : Algebra α) (h : Call) (x : Fin h.block → α) : Cell α := .mk .real (cast M (inv M h x))
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPReference
