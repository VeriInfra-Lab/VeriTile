import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Address
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import Mathlib.Tactic.FinCases
import VeriTile.Triton.Float.LogicalProgramIO

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelPublicIO
open VeriTile.Triton FP.Structural Address
noncomputable section

abbrev invocation {α : Type} (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (row : Row shape) : Invocation α where
  kernel := Source.kernel (Source.parameters st shape.columns)
  backend := M
  arguments := [⟨"eps",.real,[],fun _ => eps⟩]
  pids := fun dim => if dim=0 then row.1.val else if dim=1 then row.2.val else 0

def rowOf (shape : Shape) (j : TileIndex [shape.sequences,shape.heads,shape.columns]) : Row shape :=
  (j.1,j.2.1)

abbrev io {α : Type} (M : Algebra α) (shape : Shape) (st : Strides) (eps : α) : ProgramIO α α where
  inputs := [⟨.fp32,[shape.sequences,shape.heads,shape.columns],"X"⟩,
    ⟨.fp32,[shape.sequences,shape.heads,shape.columns],"W"⟩]
  outputs := [⟨.fp32,[shape.sequences,shape.heads,shape.columns],"Y"⟩]
  scratch := []
  buffers := ["X","W","Y"]
  read := fun i => Fin.cases (fun _ j => xOffset st (rowOf shape j) j.2.2.1.val)
    (fun k => Fin.cases (fun _ j => wOffset st (rowOf shape j) j.2.2.1.val)
      (fun z => Fin.elim0 z) k) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => yOffset st (rowOf shape j) j.2.2.1.val)
    (fun z => Fin.elim0 z) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => (Source.programs shape).map (invocation M shape st eps)
  encode := id
  observe := fun _ _ => id

def xRow {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides) (eps : α)
    (xs : (io M shape st eps).Inputs) (row : Row shape) (j : Nat) : α :=
  if h : j<shape.columns then xs 0 (row.1,row.2,⟨j,h⟩,()) else default

def wRow {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides) (eps : α)
    (xs : (io M shape st eps).Inputs) (row : Row shape) (j : Nat) : α :=
  if h : j<shape.columns then xs 1 (row.1,row.2,⟨j,h⟩,()) else default

def target {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (st : Strides) (eps : α)
    (f : (Nat → α) → (Nat → α) → Nat → α) (xs : (io M shape st eps).Inputs) :
    (io M shape st eps).Outputs :=
  fun o => Fin.cases (fun j => f (xRow M shape st eps xs (rowOf shape j))
    (wRow M shape st eps xs (rowOf shape j)) j.2.2.1.val) (fun z => Fin.elim0 z) o

end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelPublicIO

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.PairIO
open VeriTile.Triton FP.Structural Address KernelPublicIO
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvocation (M : Algebra α) (shape : Shape) (st : Strides)
    (eps : α) (row : Row shape) : Invocation α :=
  { invocation M shape st eps row with kernel := Source.candidateKernel (Source.parameters st shape.columns) }

def variant (M : Algebra α) (shape : Shape) (st : Strides) (eps : α) :
    ProgramIO.Variant (io M shape st eps) where
  schedule := fun _ => (Source.programs shape).map (candidateInvocation M shape st eps)
  observe := (io M shape st eps).observe

def original (M : Algebra α) (shape : Shape) (st : Strides) (eps : α) :=
  ProgramIO.Implementation.original (io M shape st eps) (fun _ => True)

def candidate (M : Algebra α) (shape : Shape) (st : Strides) (eps : α) :=
  ProgramIO.Implementation.ofVariant (variant M shape st eps) (fun _ => True)

end
end VeriTile.Bench.Repairs.LayernormFwdTriton.PairIO

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.LogicalIO
open VeriTile.Triton FP.Structural
noncomputable section

structure Inputs (shape : Shape) where
  x : Fin shape.sequences → Fin shape.heads → Fin shape.columns → ℝ
  weight : Fin shape.sequences → Fin shape.heads → Fin shape.columns → ℝ

def inputs (A : Algebra ℝ) (shape : Shape) (st : Strides) (eps : ℝ) : (KernelPublicIO.io A shape st eps).Inputs ≃ Inputs shape where
  toFun := fun v => ⟨fun i0 i1 i2 => v (0 : Fin 2) (i0,i1,i2,()), fun i0 i1 i2 => v (1 : Fin 2) (i0,i1,i2,())⟩
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v.x j.1 j.2.1 j.2.2.1
    | ⟨1,_⟩ => fun j => v.weight j.1 j.2.1 j.2.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    cases v
    rfl

def outputs (A : Algebra ℝ) (shape : Shape) (st : Strides) (eps : ℝ) : (KernelPublicIO.io A shape st eps).Outputs ≃ (Fin shape.sequences → Fin shape.heads → Fin shape.columns → ℝ) where
  toFun := fun v i0 i1 i2 => v (0 : Fin 1) (i0,i1,i2,())
  invFun := fun v i => match i with
    | ⟨0,_⟩ => fun j => v j.1 j.2.1 j.2.2.1
  left_inv := by
    intro v
    funext i j
    fin_cases i <;> cases j <;> rfl
  right_inv := by
    intro v
    rfl

def io (A : Algebra ℝ) (shape : Shape) (st : Strides) (eps : ℝ) : LogicalProgramIO ℝ ℝ (Inputs shape) (Fin shape.sequences → Fin shape.heads → Fin shape.columns → ℝ) where
  core := KernelPublicIO.io A shape st eps
  inputs := inputs A shape st eps
  outputs := outputs A shape st eps

end
end VeriTile.Bench.Repairs.LayernormFwdTriton.LogicalIO
