import bench.tritonbench_g.rmsnorm_implementation.Proofs.WholeStructure
import VeriTile.Triton.Float.ScalarOps

/-! The native graph boundary is explicit. The typed DSL retains split source
syntax; default compilation contracts its square-accumulate into fma.rn.f32.
This module interprets that observed graph using scalar instruction semantics,
not an experimental rewrite rule or a whole-row callback. The supplied complete
reduction tree records its ordered add.rn.f32 graph. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.NativeKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section
abbrev Word := FP.Value .fp32

/-- Scalar instruction semantics for the observed Triton 3.7.1 / SM90 profile.
Division and sqrt retain their actual PTX opcode families. IEEE software
semantics alone do not establish hardware div.full or sqrt.approx.ftz behavior. -/
structure Primitives where
  literal : ℝ → Word
  fmaRN : Word → Word → Word → Word
  addRN : Word → Word → Word
  divFull : Word → Word → Word
  sqrtApproxFTZ : Word → Word
  mulRN : Word → Word → Word

def numerics (ops : Primitives) : Numerics Word where
  literal := ops.literal
  squareAcc := fun x acc=>ops.fmaRN x x acc
  add := ops.addRN
  divideByWidth := fun x n=>ops.divFull x (ops.literal n)
  addEpsilon := fun x eps=>ops.addRN x (ops.literal eps)
  div := ops.divFull
  sqrt := ops.sqrtApproxFTZ
  mul := ops.mulRN

/-- Concrete rounded FMA/add/multiply are available without numerical
assumptions. The literal, div.full and sqrt.approx.ftz ports remain explicit. -/
def softwareScalars (cfg : FP.Config) (literal : ℝ → Word)
    (divFull : Word → Word → Word) (sqrtApproxFTZ : Word → Word) : Primitives :=
  ⟨literal,FP.fma cfg,FP.add cfg,divFull,sqrtApproxFTZ,FP.mul cfg⟩

abbrev reference (ops : Primitives) := logicalValue (numerics ops)

theorem accumulator_congr (Q : Numerics α) (n : Nat) (x y : Nat → α)
    (h : ∀ k,k<n → x k=y k) (t : Nat) : laneAccumulator Q n x t=laneAccumulator Q n y t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    funext j
    simp only [laneAccumulator,ih]
    by_cases active : t*4096+j.val<n
    · rw [if_pos active,if_pos active,h _ active]
    · rw [if_neg active,if_neg active]

theorem value_congr (Q : Numerics α) (red : Reduction) (n : Nat) (eps : ℝ)
    (x w y v : Nat → α) (k : Nat) (hk : k<n)
    (hx : ∀ j,j<n → x j=y j) (hw : ∀ j,j<n → w j=v j) :
    logicalValue Q red n eps x w k=logicalValue Q red n eps y v k := by
  unfold logicalValue logicalStd
  rw [hx k hk,hw k hk,accumulator_congr Q n x y hx]

#specNonCircular reference avoiding [executeRow,outputPass,varianceStep]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.NativeKernel
